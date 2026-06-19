const state = {
  files: [],
  prediction: null,
  loading: false,
  error: "",
  maxUploadBytes: 12 * 1024 * 1024,
  maxTotalUploadBytes: 48 * 1024 * 1024,
  maxBatchImages: 10,
};

const ACCEPTED_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp"]);
const ACCEPTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const SCHEMA_DISPLAY_NAMES = {
  clean_dots: "Clean dots",
  merged_snowman: "Merged colonies",
  streak_lines: "Streak lines",
};

const elements = {
  batchUsed: document.getElementById("batch-used"),
  confidenceRange: document.getElementById("confidence-range"),
  confidenceValue: document.getElementById("confidence-value"),
  countValue: document.getElementById("count-value"),
  detectionsBody: document.getElementById("detections-body"),
  detectionsCount: document.getElementById("detections-count"),
  detectionsPanel: document.getElementById("detections-panel"),
  downloadImage: document.getElementById("download-image"),
  downloadJson: document.getElementById("download-json"),
  dropZone: document.getElementById("drop-zone"),
  errorBanner: document.getElementById("error-banner"),
  fileInput: document.getElementById("file-input"),
  fileMeta: document.getElementById("file-meta"),
  fileName: document.getElementById("file-name"),
  form: document.getElementById("predict-form"),
  modelStatus: document.getElementById("model-status"),
  modelUsed: document.getElementById("model-used"),
  reliabilityWarning: document.getElementById("reliability-warning"),
  resultCards: document.getElementById("result-cards"),
  resultsCount: document.getElementById("results-count"),
  resultsPanel: document.getElementById("results-panel"),
  runtimeUsed: document.getElementById("runtime-used"),
  schemaUsed: document.getElementById("schema-used"),
  submitButton: document.getElementById("submit-button"),
  thresholdUsed: document.getElementById("threshold-used"),
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return "";
  }
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setError(message) {
  state.error = message || "";
  render();
}

function isAcceptedImage(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  return ACCEPTED_EXTENSIONS.has(extension) || ACCEPTED_MIME_TYPES.has(file.type);
}

function setFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    return;
  }
  if (files.length > state.maxBatchImages) {
    setError(`Too many images. Maximum batch size is ${state.maxBatchImages}.`);
    return;
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > state.maxTotalUploadBytes) {
    setError(`Batch is too large. Maximum total size is ${formatBytes(state.maxTotalUploadBytes)}.`);
    return;
  }
  const invalidFile = files.find((file) => !isAcceptedImage(file));
  if (invalidFile) {
    setError(`Unsupported file type: ${invalidFile.name}. Upload JPG, PNG, or WebP images.`);
    return;
  }
  const oversizedFile = files.find((file) => file.size > state.maxUploadBytes);
  if (oversizedFile) {
    setError(`${oversizedFile.name} is too large. Maximum single-image size is ${formatBytes(state.maxUploadBytes)}.`);
    return;
  }

  state.files = files;
  state.prediction = null;
  state.error = "";
  render();
}

function getSuccessfulResults() {
  if (!state.prediction) {
    return [];
  }
  if (Array.isArray(state.prediction.results)) {
    return state.prediction.results;
  }
  return state.prediction.ok ? [state.prediction] : [];
}

function getFailedResults() {
  return state.prediction?.errors || [];
}

function getPrimaryImageHref() {
  const first = getSuccessfulResults().find((result) => result.annotated_image_data_url || result.annotated_image_url);
  return first?.annotated_image_data_url || first?.annotated_image_url || "#";
}

function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))];
}

function formatSchema(schema) {
  return SCHEMA_DISPLAY_NAMES[schema] || String(schema || "Unknown").replaceAll("_", " ");
}

function selectedSchema(result) {
  return result?.selected_schema || result?.chosen_schema || result?.model?.schema || result?.schema;
}

function selectedModelPath(result) {
  return result?.selected_model?.path || result?.chosen_model?.path || result?.model?.path || result?.model_path;
}

function filenameForPath(path) {
  return String(path || "").split(/[\\/]/).filter(Boolean).pop() || "";
}

function specialistPathLabel(payload) {
  const specialists = payload?.model_specialists || {};
  const cleanPath =
    specialists.clean_dots?.default_path ||
    specialists.clean_dots?.path ||
    "models/apricot_clean_dot_counter_v1.pt";
  const mergedPath =
    specialists.merged_snowman?.default_path ||
    specialists.merged_snowman?.path ||
    "models/apricot_merged_colony_counter_v1.pt";
  return `Clean ${filenameForPath(cleanPath)} · Merged ${filenameForPath(mergedPath)}`;
}

function renderSummary() {
  const prediction = state.prediction;
  const results = getSuccessfulResults();
  const failures = getFailedResults();
  const totalCount = prediction?.count_total ?? prediction?.count;
  const threshold = prediction?.confidence_threshold ?? prediction?.confidence ?? results[0]?.confidence_threshold;
  const modelNames = uniqueValues(results.map(selectedModelPath));
  const schemas = uniqueValues(results.map(selectedSchema));
  const warnings = uniqueValues(results.map((result) => result.reliability_warning));

  elements.countValue.textContent = totalCount === undefined ? "--" : String(totalCount);
  elements.thresholdUsed.textContent = threshold === undefined ? "Threshold --" : `Threshold ${Number(threshold).toFixed(2)}`;
  elements.runtimeUsed.textContent = results.length === 1 && results[0].duration_ms ? `Runtime ${results[0].duration_ms} ms` : "Runtime --";
  elements.schemaUsed.textContent =
    schemas.length === 0 ? "Counted as --" : `Counted as ${schemas.length === 1 ? formatSchema(schemas[0]) : "mixed"}`;
  elements.modelUsed.textContent = modelNames.length === 0 ? "Model --" : `Model ${modelNames.length === 1 ? modelNames[0] : "mixed"}`;
  elements.batchUsed.textContent = prediction
    ? `Images ${results.length}${failures.length ? `, Failed ${failures.length}` : ""}`
    : "Images --";
  elements.reliabilityWarning.textContent = warnings[0] || "";
  elements.reliabilityWarning.classList.toggle("visible", Boolean(warnings[0]));
}

function renderUpload() {
  elements.confidenceValue.textContent = Number(elements.confidenceRange.value).toFixed(2);
  elements.submitButton.disabled = !state.files.length || state.loading;
  elements.submitButton.textContent = state.loading ? "Counting..." : state.files.length > 1 ? "Run Batch" : "Run Detection";

  if (!state.files.length) {
    elements.fileName.textContent = "Choose image";
    elements.fileMeta.textContent = `JPG, PNG, or WebP. Up to ${state.maxBatchImages} images.`;
    return;
  }
  if (state.files.length === 1) {
    elements.fileName.textContent = state.files[0].name;
    elements.fileMeta.textContent = formatBytes(state.files[0].size);
    return;
  }
  const totalBytes = state.files.reduce((sum, file) => sum + file.size, 0);
  elements.fileName.textContent = `${state.files.length} images selected`;
  elements.fileMeta.textContent = formatBytes(totalBytes);
}

function renderError() {
  elements.errorBanner.textContent = state.error;
  elements.errorBanner.classList.toggle("visible", Boolean(state.error));
}

function renderDownloads() {
  const hasResults = getSuccessfulResults().length > 0;
  const annotatedHref = getPrimaryImageHref();
  elements.downloadImage.href = annotatedHref;
  elements.downloadImage.classList.toggle("disabled", !hasResults || annotatedHref === "#");
  elements.downloadJson.disabled = !state.prediction;
}

function renderResults() {
  const results = getSuccessfulResults();
  const failures = getFailedResults();
  const hasAny = results.length > 0 || failures.length > 0;
  elements.resultsPanel.classList.toggle("hidden", !hasAny);
  elements.resultsCount.textContent = `${results.length} processed${failures.length ? `, ${failures.length} failed` : ""}`;

  const resultMarkup = results
    .map((result) => {
      const imageSrc = result.annotated_image_data_url || result.annotated_image_url || "";
      const warning = result.reliability_warning ? `<p class="warning-inline">${result.reliability_warning}</p>` : "";
      const countedAs = selectedSchema(result);
      const routedAs = result.route_schema || result.schema;
      const schemaText = routedAs && routedAs !== countedAs
        ? `${formatSchema(countedAs)} · route ${formatSchema(routedAs)}`
        : formatSchema(countedAs);
      const modelPath = selectedModelPath(result);
      return `
        <article class="result-card">
          <div class="result-card-header">
            <div>
              <h3>${result.filename || "image"}</h3>
              <span>${schemaText} · ${modelPath || "model unknown"} · ${Number(result.confidence_threshold ?? result.confidence ?? 0).toFixed(2)}</span>
            </div>
            <strong>${result.count}</strong>
          </div>
          ${warning}
          ${imageSrc ? `<img src="${imageSrc}" alt="Annotated result for ${result.filename || "image"}" />` : ""}
        </article>
      `;
    })
    .join("");

  const failureMarkup = failures
    .map(
      (failure) => `
        <article class="result-card failed-card">
          <div class="result-card-header">
            <div>
              <h3>${failure.filename || "image"}</h3>
              <span>Failed</span>
            </div>
            <strong>--</strong>
          </div>
          <p>${failure.error || "Could not process this image."}</p>
        </article>
      `,
    )
    .join("");

  elements.resultCards.innerHTML = resultMarkup + failureMarkup;
}

function renderDetections() {
  const rows = getSuccessfulResults().flatMap((result) =>
    (result.detections || []).map((detection) => ({
      filename: result.filename || "image",
      detection,
    })),
  );
  elements.detectionsPanel.classList.toggle("hidden", rows.length === 0);
  elements.detectionsCount.textContent = `${rows.length} ${rows.length === 1 ? "box" : "boxes"}`;
  elements.detectionsBody.innerHTML = rows
    .slice(0, 500)
    .map(({ filename, detection }) => {
      const box = detection.box;
      return `
        <tr>
          <td>${detection.id}</td>
          <td>${filename}</td>
          <td>${Math.round(detection.confidence * 100)}%</td>
          <td>${Math.round(box.center_x)}</td>
          <td>${Math.round(box.center_y)}</td>
          <td>${Math.round(box.width)}</td>
          <td>${Math.round(box.height)}</td>
        </tr>
      `;
    })
    .join("");
}

function render() {
  renderUpload();
  renderSummary();
  renderError();
  renderDownloads();
  renderResults();
  renderDetections();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    if (Number.isFinite(payload.max_upload_bytes)) {
      state.maxUploadBytes = payload.max_upload_bytes;
    }
    if (Number.isFinite(payload.max_total_upload_bytes)) {
      state.maxTotalUploadBytes = payload.max_total_upload_bytes;
    }
    if (Number.isFinite(payload.max_batch_images)) {
      state.maxBatchImages = payload.max_batch_images;
    }
    const specialistLabel = specialistPathLabel(payload);
    const specialists = payload.model_specialists || {};
    elements.modelStatus.title = [
      `Clean-dot: ${specialists.clean_dots?.path || "models/apricot_clean_dot_counter_v1.pt"}`,
      `Merged: ${specialists.merged_snowman?.path || "models/apricot_merged_colony_counter_v1.pt"}`,
    ].join("\n");
    if (payload.model_exists) {
      elements.modelStatus.textContent = `${payload.model_loaded ? "Loaded" : "Ready"} · ${specialistLabel}`;
      elements.modelStatus.classList.add("ready");
    } else {
      elements.modelStatus.textContent = `Weights needed · ${specialistLabel}`;
      elements.modelStatus.classList.remove("ready");
    }
    renderUpload();
  } catch {
    elements.modelStatus.textContent = "Service unavailable";
    elements.modelStatus.classList.remove("ready");
  }
}

async function submitPrediction() {
  if (!state.files.length || state.loading) {
    return;
  }

  state.loading = true;
  state.error = "";
  state.prediction = null;
  render();

  const isBatch = state.files.length > 1;
  const formData = new FormData();
  state.files.forEach((file) => formData.append(isBatch ? "images" : "file", file));
  formData.append("confidence", elements.confidenceRange.value);
  formData.append("include_images", "true");

  try {
    const response = await fetch(isBatch ? "/api/predict-batch" : "/api/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `Prediction failed with status ${response.status}.`);
    }
    state.prediction = payload;
  } catch (error) {
    setError(error.message || "Prediction failed.");
  } finally {
    state.loading = false;
    render();
    checkHealth();
  }
}

elements.fileInput.addEventListener("change", (event) => {
  setFiles(event.target.files);
});

elements.confidenceRange.addEventListener("input", render);

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPrediction();
});

elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
});

elements.dropZone.addEventListener("dragleave", () => {
  elements.dropZone.classList.remove("dragging");
});

elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  setFiles(event.dataTransfer.files);
});

elements.downloadJson.addEventListener("click", () => {
  if (!state.prediction) {
    return;
  }
  const blob = new Blob([JSON.stringify(state.prediction, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "apricot-detections.json";
  anchor.click();
  URL.revokeObjectURL(url);
});

render();
checkHealth();
