const state = {
  analysis: null,
  fileName: "",
  imageUrl: "",
  threshold: 0.45,
  showLabels: true,
  showRejected: false,
  analyzing: false,
  stageIndex: 0,
  selectedId: null,
  manualColonies: [],
  removedIds: new Set(),
  corrections: { added: 0, removed: 0 },
  error: "",
};

const LOADING_STEPS = [
  "Uploading plate image...",
  "Detecting dish circle...",
  "Finding blob candidates...",
  "Filtering likely labels...",
  "Scoring colonies...",
];

const elements = {
  downloadJson: document.getElementById("download-json"),
  errorBanner: document.getElementById("error-banner"),
  fileInput: document.getElementById("file-input"),
  fileMeta: document.getElementById("file-meta"),
  heroPanel: document.getElementById("hero-panel"),
  labelList: document.getElementById("label-list"),
  legendColonies: document.getElementById("legend-colonies"),
  legendLabels: document.getElementById("legend-labels"),
  legendRejected: document.getElementById("legend-rejected"),
  loadingPanel: document.getElementById("loading-panel"),
  loadingText: document.getElementById("loading-text"),
  pipelineSteps: document.getElementById("pipeline-steps"),
  progressBar: document.getElementById("progress-bar"),
  removeSelected: document.getElementById("remove-selected"),
  resetSession: document.getElementById("reset-session"),
  resultsLayout: document.getElementById("results-layout"),
  selectionText: document.getElementById("selection-text"),
  statsCandidates: document.getElementById("stat-candidates"),
  statsColonies: document.getElementById("stat-colonies"),
  statsConfidence: document.getElementById("stat-confidence"),
  statsCorrections: document.getElementById("stat-corrections"),
  statsGrid: document.getElementById("stats-grid"),
  statsLabels: document.getElementById("stat-labels"),
  svg: document.getElementById("analysis-svg"),
  tableBody: document.getElementById("colony-table-body"),
  thresholdRange: document.getElementById("threshold-range"),
  thresholdValue: document.getElementById("threshold-value"),
  toggleLabels: document.getElementById("toggle-labels"),
  toggleRejected: document.getElementById("toggle-rejected"),
  uploadButton: document.getElementById("upload-button"),
};

let loadingTimer = null;

function revokeImageUrl() {
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = "";
  }
}

function setError(message) {
  state.error = message || "";
  render();
}

function resetCorrections() {
  state.manualColonies = [];
  state.removedIds = new Set();
  state.corrections = { added: 0, removed: 0 };
  state.selectedId = null;
}

function fullReset() {
  state.analysis = null;
  state.fileName = "";
  state.analyzing = false;
  state.stageIndex = 0;
  resetCorrections();
  setError("");
  revokeImageUrl();
  elements.fileInput.value = "";
  render();
}

function getBaseColonies() {
  if (!state.analysis) {
    return [];
  }
  return state.analysis.colonies.filter((item) => !state.removedIds.has(item.id));
}

function getVisibleColonies() {
  return [...getBaseColonies(), ...state.manualColonies].filter((item) => item.conf >= state.threshold);
}

function getAverageConfidence() {
  const visible = getVisibleColonies();
  if (visible.length === 0) {
    return 0;
  }
  return visible.reduce((sum, item) => sum + item.conf, 0) / visible.length;
}

function getSelectedColony() {
  return getVisibleColonies().find((item) => item.id === state.selectedId) || null;
}

function startLoading() {
  state.analyzing = true;
  state.stageIndex = 0;
  if (loadingTimer) {
    window.clearInterval(loadingTimer);
  }
  loadingTimer = window.setInterval(() => {
    if (state.stageIndex < LOADING_STEPS.length - 1) {
      state.stageIndex += 1;
      render();
    }
  }, 450);
}

function stopLoading() {
  state.analyzing = false;
  state.stageIndex = LOADING_STEPS.length - 1;
  if (loadingTimer) {
    window.clearInterval(loadingTimer);
    loadingTimer = null;
  }
}

function renderPipeline() {
  const steps = [];
  if (state.analyzing) {
    LOADING_STEPS.forEach((label, index) => {
      let status = "waiting";
      if (index < state.stageIndex) {
        status = "done";
      } else if (index === state.stageIndex) {
        status = "active";
      }
      steps.push({ label, status, detail: "" });
    });
  } else if (state.analysis) {
    state.analysis.pipeline_steps.forEach((item) => {
      steps.push({ label: item.label, status: item.status, detail: item.detail || "" });
    });
  } else {
    [
      "Dish Detection",
      "Candidate Detection",
      "Label Filter",
      "Colony Scoring",
      "Manual Review",
    ].forEach((label) => steps.push({ label, status: "waiting", detail: "" }));
  }

  elements.pipelineSteps.innerHTML = steps
    .map(
      (step) => `
        <div class="step ${step.status}">
          <span>${step.status === "done" ? "✓" : step.status === "active" ? "..." : "o"}</span>
          <span>${step.label}</span>
          <span class="detail">${step.detail || ""}</span>
        </div>
      `,
    )
    .join("");
}

function renderStats() {
  const hasAnalysis = Boolean(state.analysis);
  const visibleColonies = getVisibleColonies();
  const summary = state.analysis?.summary || {
    candidate_count: 0,
    label_count: 0,
    rejected_count: 0,
  };

  elements.statsGrid.classList.toggle("hidden", !hasAnalysis);
  elements.resultsLayout.classList.toggle("hidden", !hasAnalysis);
  elements.downloadJson.disabled = !hasAnalysis;
  elements.heroPanel.classList.toggle("hidden", hasAnalysis || state.analyzing);

  elements.statsColonies.textContent = String(visibleColonies.length);
  elements.statsLabels.textContent = String(summary.label_count);
  elements.statsCandidates.textContent = String(summary.candidate_count);
  elements.statsCorrections.textContent = String(state.corrections.added + state.corrections.removed);
  elements.statsConfidence.textContent = `${Math.round(getAverageConfidence() * 100)}%`;

  elements.legendColonies.textContent = String(visibleColonies.length);
  elements.legendLabels.textContent = String(summary.label_count);
  elements.legendRejected.textContent = String(summary.rejected_count);
}

function renderError() {
  elements.errorBanner.textContent = state.error;
  elements.errorBanner.classList.toggle("hidden", !state.error);
}

function renderLoading() {
  elements.loadingPanel.classList.toggle("hidden", !state.analyzing);
  if (!state.analyzing) {
    elements.progressBar.style.width = "0%";
    return;
  }
  elements.loadingText.textContent = LOADING_STEPS[state.stageIndex];
  elements.progressBar.style.width = `${((state.stageIndex + 1) / LOADING_STEPS.length) * 100}%`;
}

function renderTable() {
  const selectedId = state.selectedId;
  const visibleColonies = getVisibleColonies();
  elements.tableBody.innerHTML = visibleColonies
    .slice(0, 150)
    .map(
      (item, index) => `
        <tr data-select-id="${item.id}" class="${selectedId === item.id ? "selected-row" : ""}">
          <td>${index + 1}</td>
          <td>${Math.round(item.x)}</td>
          <td>${Math.round(item.y)}</td>
          <td>${Math.round(item.conf * 100)}%</td>
          <td>${Math.round(item.area)}</td>
          <td>${Math.round(item.label_score * 100)}%</td>
        </tr>
      `,
    )
    .join("");
}

function renderLabelList() {
  const labels = state.analysis?.labels || [];
  elements.labelList.innerHTML = labels
    .slice(0, 40)
    .map(
      (item) => `
        <div class="label-card">
          <strong>${Math.round(item.label_score * 100)}% label score</strong>
          <div class="muted">x=${Math.round(item.x)} y=${Math.round(item.y)} r=${item.r.toFixed(1)}</div>
          <div class="muted">contrast ${item.local_contrast.toFixed(0)} | circularity ${item.circularity.toFixed(2)}</div>
        </div>
      `,
    )
    .join("");
}

function svgPoint(event) {
  const svg = elements.svg;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function renderSvg() {
  const analysis = state.analysis;
  if (!analysis) {
    elements.svg.setAttribute("viewBox", "0 0 100 100");
    elements.svg.innerHTML = "";
    elements.fileMeta.textContent = "Upload an image to begin.";
    return;
  }

  const visibleColonies = getVisibleColonies();
  const labels = state.showLabels ? analysis.labels : [];
  const rejected = state.showRejected ? analysis.rejected : [];
  const dish = analysis.dish;
  const image = analysis.image;

  elements.svg.setAttribute("viewBox", `0 0 ${image.width} ${image.height}`);
  elements.fileMeta.textContent = `${state.fileName} | ${image.width}x${image.height} | click inside the dish to add a colony`;

  const rejectedMarkup = rejected
    .map(
      (item) => `
        <circle cx="${item.x}" cy="${item.y}" r="${Math.max(4, item.r)}" fill="rgba(102, 200, 255, 0.08)" stroke="rgba(102, 200, 255, 0.9)" stroke-width="1.5" stroke-dasharray="5 4"></circle>
      `,
    )
    .join("");

  const labelMarkup = labels
    .map(
      (item) => `
        <g>
          <circle cx="${item.x}" cy="${item.y}" r="${Math.max(4, item.r)}" fill="rgba(255, 113, 109, 0.10)" stroke="rgba(255, 113, 109, 0.95)" stroke-width="2"></circle>
          <circle cx="${item.x}" cy="${item.y}" r="1.7" fill="rgba(255, 113, 109, 1)"></circle>
        </g>
      `,
    )
    .join("");

  const colonyMarkup = visibleColonies
    .map((item) => {
      const selected = item.id === state.selectedId;
      const selectionRing = selected
        ? `<circle cx="${item.x}" cy="${item.y}" r="${item.r + 6}" fill="none" stroke="#ffffff" stroke-width="2" stroke-dasharray="5 4"></circle>`
        : "";
      return `
        <g data-colony-id="${item.id}">
          ${selectionRing}
          <circle cx="${item.x}" cy="${item.y}" r="${Math.max(4, item.r)}" fill="rgba(63, 224, 143, 0.10)" stroke="rgba(63, 224, 143, 0.95)" stroke-width="${selected ? 3 : 2}"></circle>
          <circle cx="${item.x}" cy="${item.y}" r="1.8" fill="rgba(63, 224, 143, 1)"></circle>
        </g>
      `;
    })
    .join("");

  elements.svg.innerHTML = `
    <image href="${state.imageUrl}" x="0" y="0" width="${image.width}" height="${image.height}" preserveAspectRatio="xMidYMid meet"></image>
    <circle cx="${dish.x}" cy="${dish.y}" r="${dish.radius}" fill="none" stroke="rgba(215, 179, 86, 0.95)" stroke-width="3"></circle>
    ${rejectedMarkup}
    ${labelMarkup}
    ${colonyMarkup}
  `;
}

function renderSelection() {
  const selected = getSelectedColony();
  elements.removeSelected.disabled = !selected;
  if (!selected) {
    elements.selectionText.textContent = "Select a colony marker to remove it, or click inside the dish to add one.";
    return;
  }
  elements.selectionText.textContent = `Selected ${selected.id} at (${Math.round(selected.x)}, ${Math.round(selected.y)}) with ${Math.round(selected.conf * 100)}% confidence.`;
}

function render() {
  elements.thresholdRange.value = String(Math.round(state.threshold * 100));
  elements.thresholdValue.textContent = `${Math.round(state.threshold * 100)}%`;
  elements.toggleLabels.checked = state.showLabels;
  elements.toggleRejected.checked = state.showRejected;
  renderPipeline();
  renderLoading();
  renderError();
  renderStats();
  renderSvg();
  renderTable();
  renderLabelList();
  renderSelection();
}

async function analyzeFile(file) {
  startLoading();
  setError("");
  resetCorrections();
  state.analysis = null;
  render();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Analysis failed.");
    }
    state.analysis = payload.analysis;
  } catch (error) {
    setError(error.message || "Analysis failed.");
  } finally {
    stopLoading();
    render();
  }
}

function handleUpload(file) {
  if (!file) {
    return;
  }
  revokeImageUrl();
  state.imageUrl = URL.createObjectURL(file);
  state.fileName = file.name;
  analyzeFile(file);
}

elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", (event) => handleUpload(event.target.files?.[0]));

elements.thresholdRange.addEventListener("input", (event) => {
  state.threshold = Number(event.target.value) / 100;
  render();
});

elements.toggleLabels.addEventListener("change", (event) => {
  state.showLabels = event.target.checked;
  render();
});

elements.toggleRejected.addEventListener("change", (event) => {
  state.showRejected = event.target.checked;
  render();
});

elements.resetSession.addEventListener("click", () => fullReset());

elements.removeSelected.addEventListener("click", () => {
  const selected = getSelectedColony();
  if (!selected) {
    return;
  }
  if (selected.id.startsWith("manual-")) {
    state.manualColonies = state.manualColonies.filter((item) => item.id !== selected.id);
  } else {
    state.removedIds.add(selected.id);
  }
  state.selectedId = null;
  state.corrections.removed += 1;
  render();
});

elements.downloadJson.addEventListener("click", () => {
  if (!state.analysis) {
    return;
  }
  const payload = JSON.stringify(state.analysis, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${state.fileName.replace(/\.[^.]+$/, "") || "gold-experience"}.analysis.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});

elements.svg.addEventListener("click", (event) => {
  if (!state.analysis) {
    return;
  }

  const colonyGroup = event.target.closest("[data-colony-id]");
  if (colonyGroup) {
    state.selectedId = colonyGroup.getAttribute("data-colony-id");
    render();
    return;
  }

  if (event.target.tagName.toLowerCase() !== "svg" && event.target.tagName.toLowerCase() !== "image") {
    return;
  }

  const point = svgPoint(event);
  const { dish } = state.analysis;
  const dx = point.x - dish.x;
  const dy = point.y - dish.y;
  if (Math.sqrt(dx * dx + dy * dy) > dish.radius) {
    return;
  }

  const manualColony = {
    id: `manual-${Date.now()}`,
    candidate_id: -1,
    kind: "manual",
    x: point.x,
    y: point.y,
    r: 7,
    bbox: [Math.round(point.x - 7), Math.round(point.y - 7), 14, 14],
    area: 154,
    size: "medium",
    conf: 1.0,
    colony_score: 1.0,
    label_score: 0.0,
    rim_margin: 0.0,
    circularity: 1.0,
    solidity: 1.0,
    local_contrast: 0.0,
    edge_strength: 0.0,
  };
  state.manualColonies.push(manualColony);
  state.corrections.added += 1;
  state.selectedId = manualColony.id;
  render();
});

elements.tableBody.addEventListener("click", (event) => {
  const row = event.target.closest("[data-select-id]");
  if (!row) {
    return;
  }
  state.selectedId = row.getAttribute("data-select-id");
  render();
});

window.addEventListener("beforeunload", () => revokeImageUrl());

render();
