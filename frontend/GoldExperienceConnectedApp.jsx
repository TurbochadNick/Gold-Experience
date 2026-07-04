import { useEffect, useRef, useState } from "react";

const GOLD = "#D4A843";
const GOLD_LIGHT = "#F2D477";
const GREEN = "#00E676";
const RED = "#FF6B6B";
const BLUE = "#4FC3F7";
const DARK = "#0A1628";
const CARD = "#162240";
const SURFACE = "#111D33";
const API_BASE =
  (typeof window !== "undefined" && window.GOLD_EXPERIENCE_API_URL) ||
  "http://127.0.0.1:8000";

const LOADING_STEPS = [
  "Uploading plate image...",
  "Detecting dish circle...",
  "Finding blob candidates...",
  "Filtering likely labels...",
  "Scoring colonies...",
];

function StatCard({ value, label, color = GOLD_LIGHT, icon }) {
  return (
    <div
      style={{
        background: `linear-gradient(145deg, ${CARD}, ${SURFACE})`,
        border: "1px solid rgba(212,168,67,0.18)",
        borderRadius: 12,
        padding: "16px 14px",
        textAlign: "center",
        flex: 1,
        minWidth: 100,
      }}
    >
      {icon && <div style={{ fontSize: "0.9rem", marginBottom: 4 }}>{icon}</div>}
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "1.8rem",
          fontWeight: 700,
          color,
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: "0.6rem",
          color: "#8892A8",
          textTransform: "uppercase",
          letterSpacing: 1.5,
          marginTop: 5,
        }}
      >
        {label}
      </div>
    </div>
  );
}

function PipelineStep({ label, status, detail }) {
  const colors = { done: GREEN, active: GOLD_LIGHT, waiting: "#546E7A" };
  const icons = { done: "✓", active: "⟳", waiting: "○" };
  const color = colors[status] || colors.waiting;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "7px 12px",
        borderRadius: 6,
        marginBottom: 3,
        background:
          status === "done"
            ? "rgba(0,230,118,0.06)"
            : status === "active"
              ? "rgba(212,168,67,0.08)"
              : "rgba(84,110,122,0.05)",
        borderLeft: `3px solid ${color}`,
      }}
    >
      <span
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "0.7rem",
          color,
          fontWeight: 600,
        }}
      >
        {icons[status]}
      </span>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "0.72rem", color }}>
        {label}
      </span>
      {detail && (
        <span style={{ marginLeft: "auto", fontSize: "0.65rem", color: "#546E7A" }}>{detail}</span>
      )}
    </div>
  );
}

function MarkerLegend({ color, label, count }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "5px 10px",
        background: `${color}12`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 5,
        marginBottom: 3,
      }}
    >
      <div style={{ width: 9, height: 9, borderRadius: "50%", background: color }} />
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "0.72rem", color: "#E8EAF0" }}>
        {label}: <strong>{count}</strong>
      </span>
    </div>
  );
}

function AnalysisCanvas({
  imageUrl,
  analysis,
  colonies,
  selectedId,
  onSelect,
  onAdd,
  showLabels,
  showRejected,
}) {
  const svgRef = useRef(null);
  const { image, dish, labels, rejected } = analysis;

  const toSvgCoords = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * image.width,
      y: ((event.clientY - rect.top) / rect.height) * image.height,
    };
  };

  const handleBackgroundClick = (event) => {
    if (!svgRef.current) {
      return;
    }
    const target = event.target;
    if (target.tagName !== "svg" && !target.classList.contains("analysis-image")) {
      return;
    }
    const coords = toSvgCoords(event);
    const dx = coords.x - dish.x;
    const dy = coords.y - dish.y;
    if (Math.sqrt(dx * dx + dy * dy) <= dish.radius) {
      onAdd(coords.x, coords.y);
    }
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${image.width} ${image.height}`}
      style={{ width: "100%", maxWidth: 720, cursor: "crosshair", filter: "drop-shadow(0 6px 30px rgba(0,0,0,0.45))" }}
      onClick={handleBackgroundClick}
    >
      <image
        className="analysis-image"
        href={imageUrl}
        x="0"
        y="0"
        width={image.width}
        height={image.height}
        preserveAspectRatio="xMidYMid meet"
      />

      <circle cx={dish.x} cy={dish.y} r={dish.radius} fill="none" stroke="rgba(212,168,67,0.9)" strokeWidth={3} />

      {showRejected &&
        rejected.map((item) => (
          <circle
            key={item.id}
            cx={item.x}
            cy={item.y}
            r={Math.max(4, item.r)}
            fill="rgba(79,195,247,0.08)"
            stroke="rgba(79,195,247,0.9)"
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
        ))}

      {showLabels &&
        labels.map((item) => (
          <g key={item.id}>
            <circle
              cx={item.x}
              cy={item.y}
              r={Math.max(4, item.r)}
              fill="rgba(255,107,107,0.12)"
              stroke={RED}
              strokeWidth={2}
            />
            <circle cx={item.x} cy={item.y} r={1.5} fill={RED} />
          </g>
        ))}

      {colonies.map((item) => {
        const selected = item.id === selectedId;
        return (
          <g
            key={item.id}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(item.id);
            }}
            style={{ cursor: "pointer" }}
          >
            {selected && (
              <circle
                cx={item.x}
                cy={item.y}
                r={item.r + 6}
                fill="none"
                stroke="#fff"
                strokeWidth={2}
                strokeDasharray="5 3"
              >
                <animate attributeName="stroke-dashoffset" from="0" to="16" dur="0.9s" repeatCount="indefinite" />
              </circle>
            )}
            <circle
              cx={item.x}
              cy={item.y}
              r={Math.max(4, item.r)}
              fill="rgba(0,230,118,0.12)"
              stroke={GREEN}
              strokeWidth={selected ? 3 : 2}
            />
            <circle cx={item.x} cy={item.y} r={1.8} fill={GREEN} />
          </g>
        );
      })}
    </svg>
  );
}

export default function ApricotConnectedApp() {
  const [analysis, setAnalysis] = useState(null);
  const [manualColonies, setManualColonies] = useState([]);
  const [showLabels, setShowLabels] = useState(true);
  const [showRejected, setShowRejected] = useState(false);
  const [threshold, setThreshold] = useState(0.45);
  const [selectedId, setSelectedId] = useState(null);
  const [corrections, setCorrections] = useState({ added: 0, removed: 0 });
  const [analyzing, setAnalyzing] = useState(false);
  const [stage, setStage] = useState(0);
  const [tab, setTab] = useState("detect");
  const [error, setError] = useState("");
  const [uploadedName, setUploadedName] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const fileRef = useRef(null);

  useEffect(() => {
    if (!analyzing) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setStage((current) => Math.min(current + 1, LOADING_STEPS.length - 1));
    }, 500);
    return () => window.clearInterval(timer);
  }, [analyzing]);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const colonies = [
    ...(analysis?.colonies || []),
    ...manualColonies,
  ];
  const filteredColonies = colonies.filter((item) => item.conf >= threshold);
  const summary = analysis?.summary || {
    candidate_count: 0,
    colony_count: 0,
    label_count: 0,
    rejected_count: 0,
    average_confidence: 0,
  };
  const avgConfidence =
    filteredColonies.length > 0
      ? filteredColonies.reduce((total, item) => total + item.conf, 0) / filteredColonies.length
      : 0;

  const runAnalysis = async (file) => {
    setAnalyzing(true);
    setStage(0);
    setError("");
    setSelectedId(null);
    setManualColonies([]);
    setCorrections({ added: 0, removed: 0 });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Analysis failed");
      }
      setAnalysis(payload.analysis);
      setTab("detect");
    } catch (requestError) {
      setAnalysis(null);
      setError(requestError.message || "Unable to analyze image.");
    } finally {
      setAnalyzing(false);
      setStage(LOADING_STEPS.length - 1);
    }
  };

  const handleUpload = async (event) => {
    const file = event?.target?.files?.[0];
    if (!file) {
      return;
    }
    setUploadedName(file.name);
    setImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return URL.createObjectURL(file);
    });
    await runAnalysis(file);
  };

  const handleRemove = () => {
    if (!selectedId) {
      return;
    }
    if (selectedId.startsWith("manual-")) {
      setManualColonies((current) => current.filter((item) => item.id !== selectedId));
    } else {
      setAnalysis((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          colonies: current.colonies.filter((item) => item.id !== selectedId),
          summary: {
            ...current.summary,
            colony_count: Math.max(0, current.summary.colony_count - 1),
          },
        };
      });
    }
    setCorrections((current) => ({ ...current, removed: current.removed + 1 }));
    setSelectedId(null);
  };

  const handleAdd = (x, y) => {
    const item = {
      id: `manual-${Date.now()}`,
      candidate_id: -1,
      kind: "manual",
      x,
      y,
      r: 7,
      bbox: [Math.round(x - 7), Math.round(y - 7), 14, 14],
      area: 154,
      size: "medium",
      conf: 1.0,
      colony_score: 1.0,
      label_score: 0.0,
      rim_margin: 0,
      circularity: 1.0,
      solidity: 1.0,
      local_contrast: 0,
      edge_strength: 0,
    };
    setManualColonies((current) => [...current, item]);
    setCorrections((current) => ({ ...current, added: current.added + 1 }));
  };

  const requiem = () => {
    setAnalysis(null);
    setManualColonies([]);
    setCorrections({ added: 0, removed: 0 });
    setAnalyzing(false);
    setSelectedId(null);
    setStage(0);
    setTab("detect");
    setUploadedName("");
    setThreshold(0.45);
    setShowLabels(true);
    setShowRejected(false);
    setError("");
    setImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return "";
    });
    if (fileRef.current) {
      fileRef.current.value = "";
    }
  };

  const steps = analysis?.pipeline_steps || [];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: `linear-gradient(160deg, #070D1A 0%, ${DARK} 40%, #0F1F3D 100%)`,
        color: "#E8EAF0",
        fontFamily: "'Outfit', 'Segoe UI', sans-serif",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap"
        rel="stylesheet"
      />

      <div style={{ display: "flex", minHeight: "100vh" }}>
        <div
          style={{
            width: 300,
            flexShrink: 0,
            background: "linear-gradient(180deg, #0C1829, #091322)",
            borderRight: "1px solid rgba(212,168,67,0.12)",
            padding: "20px 16px",
            display: "flex",
            flexDirection: "column",
            overflowY: "auto",
          }}
        >
          <div style={{ textAlign: "center", marginBottom: 20 }}>
            <div style={{ fontSize: "1.6rem" }}>🧬</div>
            <div
              style={{
                fontSize: "1.2rem",
                fontWeight: 900,
                background: `linear-gradient(135deg, ${GOLD_LIGHT}, ${GOLD})`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              APRICOT
            </div>
            <div
              style={{
                fontFamily: "'JetBrains Mono'",
                fontSize: "0.55rem",
                color: "#546E7A",
                letterSpacing: 2,
                textTransform: "uppercase",
              }}
            >
              Apricot Colony Counter v1
            </div>
          </div>

          <div style={{ height: 1, background: "rgba(212,168,67,0.12)", margin: "0 0 16px" }} />

          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: GOLD, marginBottom: 10 }}>
              Detection Settings
            </div>
            <label style={{ fontSize: "0.68rem", color: "#8892A8" }}>
              Colony confidence: {(threshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="10"
              max="95"
              value={threshold * 100}
              onChange={(event) => setThreshold(Number(event.target.value) / 100)}
              style={{ width: "100%", accentColor: GOLD, marginTop: 2 }}
            />
            <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: "0.72rem", cursor: "pointer", marginTop: 8, color: "#C0C8D8" }}>
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(event) => setShowLabels(event.target.checked)}
                style={{ accentColor: GOLD }}
              />
              Show filtered labels
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: "0.72rem", cursor: "pointer", marginTop: 8, color: "#C0C8D8" }}>
              <input
                type="checkbox"
                checked={showRejected}
                onChange={(event) => setShowRejected(event.target.checked)}
                style={{ accentColor: GOLD }}
              />
              Show rejected candidates
            </label>
          </div>

          <div style={{ height: 1, background: "rgba(212,168,67,0.12)", margin: "0 0 16px" }} />

          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: GOLD, marginBottom: 10 }}>
              Pipeline Status
            </div>
            {analysis && !analyzing ? (
              <>
                {steps.map((item) => (
                  <PipelineStep key={item.key} label={item.label} status={item.status} detail={item.detail} />
                ))}
                <div style={{ marginTop: 6, fontSize: "0.65rem", color: GREEN }}>
                  Engine: Apricot API
                </div>
              </>
            ) : analyzing ? (
              LOADING_STEPS.map((label, index) => (
                <PipelineStep
                  key={label}
                  label={label}
                  status={index < stage ? "done" : index === stage ? "active" : "waiting"}
                />
              ))
            ) : (
              <>
                <PipelineStep label="Dish Detection" status="waiting" />
                <PipelineStep label="Candidate Detection" status="waiting" />
                <PipelineStep label="Label Filter" status="waiting" />
                <PipelineStep label="Colony Scoring" status="waiting" />
                <PipelineStep label="Manual Review" status="waiting" />
              </>
            )}
          </div>

          {analysis && !analyzing && (
            <>
              <div style={{ height: 1, background: "rgba(212,168,67,0.12)", margin: "0 0 16px" }} />
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: "0.78rem", fontWeight: 700, color: GOLD, marginBottom: 10 }}>
                  Detection Layers
                </div>
                <MarkerLegend color={GREEN} label="Colonies" count={summary.colony_count + manualColonies.length} />
                <MarkerLegend color={RED} label="Labels" count={summary.label_count} />
                <MarkerLegend color={BLUE} label="Rejected" count={summary.rejected_count} />
              </div>
            </>
          )}

          <div style={{ flex: 1 }} />

          <div style={{ height: 1, background: "rgba(212,168,67,0.12)", margin: "0 0 12px" }} />

          <button
            onClick={requiem}
            style={{
              width: "100%",
              padding: "11px",
              background: "linear-gradient(135deg, #1a0a0a, #2d1515)",
              color: RED,
              border: "1px solid rgba(255,107,107,0.25)",
              borderRadius: 10,
              cursor: "pointer",
              fontWeight: 700,
              fontSize: "0.85rem",
              fontFamily: "'Outfit', sans-serif",
            }}
          >
            Reset Session
          </button>
        </div>

        <div style={{ flex: 1, padding: "24px 28px", overflowY: "auto" }}>
          <div style={{ textAlign: "center", marginBottom: 24 }}>
            <div
              style={{
                fontSize: "2.4rem",
                fontWeight: 900,
                background: `linear-gradient(135deg, ${GOLD_LIGHT}, ${GOLD}, #A67C2E)`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                letterSpacing: -1,
              }}
            >
              「APRICOT」
            </div>
            <div
              style={{
                fontFamily: "'JetBrains Mono'",
                fontSize: "0.68rem",
                color: "#8892A8",
                letterSpacing: 3,
                textTransform: "uppercase",
                marginTop: 2,
              }}
            >
              Explainable Colony Detection + Label Rejection
            </div>
          </div>

          {!analysis && !analyzing ? (
            <div style={{ maxWidth: 620, margin: "40px auto" }}>
              <input ref={fileRef} type="file" accept="image/*" onChange={handleUpload} style={{ display: "none" }} />
              <div
                onClick={() => fileRef.current?.click()}
                style={{
                  background: "linear-gradient(145deg, rgba(22,34,64,0.8), rgba(17,29,51,0.8))",
                  border: "2px dashed rgba(212,168,67,0.25)",
                  borderRadius: 18,
                  padding: "56px 28px",
                  textAlign: "center",
                  cursor: "pointer",
                }}
              >
                <div style={{ fontSize: "3.5rem", marginBottom: 14 }}>🧫</div>
                <div style={{ fontSize: "1.15rem", color: GOLD, fontWeight: 700, marginBottom: 8 }}>
                  Upload Petri Dish Image
                </div>
                <div style={{ fontSize: "0.85rem", color: "#8892A8", maxWidth: 420, margin: "0 auto" }}>
                  The app sends the image to your local Apricot API, detects the plate,
                  filters labels, and returns colony candidates for review.
                </div>
                <div style={{ marginTop: 14, fontFamily: "'JetBrains Mono'", fontSize: "0.65rem", color: "#546E7A" }}>
                  API: {API_BASE}
                </div>
              </div>

              {error && (
                <div
                  style={{
                    marginTop: 14,
                    padding: "12px 14px",
                    background: "rgba(255,107,107,0.08)",
                    border: "1px solid rgba(255,107,107,0.2)",
                    borderRadius: 10,
                    color: "#FFC1C1",
                    fontSize: "0.78rem",
                  }}
                >
                  {error}
                </div>
              )}
            </div>
          ) : analyzing ? (
            <div style={{ textAlign: "center", padding: "60px 0" }}>
              <div style={{ fontFamily: "'JetBrains Mono'", fontSize: "0.95rem", color: GOLD_LIGHT, marginBottom: 20 }}>
                {LOADING_STEPS[stage]}
              </div>
              <div style={{ width: 260, height: 5, background: CARD, borderRadius: 3, margin: "0 auto", overflow: "hidden" }}>
                <div
                  style={{
                    width: `${((stage + 1) / LOADING_STEPS.length) * 100}%`,
                    height: "100%",
                    background: `linear-gradient(90deg, ${GOLD}, ${GREEN})`,
                    borderRadius: 3,
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
            </div>
          ) : (
            <>
              <div style={{ display: "flex", gap: 12, marginBottom: 22, maxWidth: 860, margin: "0 auto 22px", flexWrap: "wrap" }}>
                <StatCard icon="🧫" value={filteredColonies.length} label="Visible Colonies" />
                <StatCard icon="🚫" value={summary.label_count} label="Labels Filtered" color={RED} />
                <StatCard icon="🔎" value={summary.candidate_count} label="Candidates" color={BLUE} />
                <StatCard icon="🖊️" value={corrections.added + corrections.removed} label="Corrections" />
                <StatCard icon="📊" value={`${(avgConfidence * 100).toFixed(0)}%`} label="Avg Confidence" color={GREEN} />
              </div>

              <div style={{ display: "flex", gap: 4, marginBottom: 20, maxWidth: 700, margin: "0 auto 20px", background: CARD, borderRadius: 10, padding: 4 }}>
                {[["detect", "Detection"], ["labels", "Labels"], ["data", "Data"]].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setTab(key)}
                    style={{
                      flex: 1,
                      padding: "10px 8px",
                      background: tab === key ? `linear-gradient(135deg, #A67C2E, ${GOLD})` : "transparent",
                      color: tab === key ? DARK : "#8892A8",
                      border: "none",
                      borderRadius: 8,
                      fontWeight: 700,
                      fontSize: "0.78rem",
                      cursor: "pointer",
                      fontFamily: "'Outfit', sans-serif",
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {tab === "detect" && (
                <div style={{ display: "flex", gap: 20, maxWidth: 1120, margin: "0 auto", flexWrap: "wrap" }}>
                  <div style={{ flex: "1 1 620px", minWidth: 320 }}>
                    <AnalysisCanvas
                      imageUrl={imageUrl}
                      analysis={analysis}
                      colonies={filteredColonies}
                      selectedId={selectedId}
                      onSelect={setSelectedId}
                      onAdd={handleAdd}
                      showLabels={showLabels}
                      showRejected={showRejected}
                    />
                    <div style={{ textAlign: "center", marginTop: 8, fontSize: "0.68rem", color: "#546E7A" }}>
                      Click inside the dish to add a colony. Select a colony to remove it. File: {uploadedName}
                    </div>
                  </div>
                  <div style={{ flex: "0 0 260px", display: "flex", flexDirection: "column", gap: 12 }}>
                    <div style={{ background: CARD, border: "1px solid rgba(212,168,67,0.12)", borderRadius: 10, padding: 16 }}>
                      <div style={{ fontSize: "0.78rem", fontWeight: 700, color: GOLD, marginBottom: 10 }}>
                        Manual Corrections
                      </div>
                      {selectedId ? (
                        <>
                          <div style={{ fontSize: "0.7rem", color: "#C0C8D8", marginBottom: 8 }}>
                            Selected: <code style={{ color: GOLD_LIGHT }}>{selectedId}</code>
                          </div>
                          <button
                            onClick={handleRemove}
                            style={{
                              width: "100%",
                              padding: "10px",
                              background: "linear-gradient(135deg, #3D1515, #2D1010)",
                              color: RED,
                              border: "1px solid rgba(255,107,107,0.3)",
                              borderRadius: 8,
                              cursor: "pointer",
                              fontWeight: 600,
                              fontSize: "0.8rem",
                              fontFamily: "'Outfit'",
                            }}
                          >
                            Remove Selected
                          </button>
                          <button
                            onClick={() => setSelectedId(null)}
                            style={{
                              width: "100%",
                              padding: "8px",
                              background: "transparent",
                              color: "#546E7A",
                              border: "1px solid rgba(84,110,122,0.3)",
                              borderRadius: 8,
                              cursor: "pointer",
                              fontSize: "0.72rem",
                              marginTop: 6,
                              fontFamily: "'Outfit'",
                            }}
                          >
                            Deselect
                          </button>
                        </>
                      ) : (
                        <div style={{ fontSize: "0.7rem", color: "#546E7A", textAlign: "center", padding: 12 }}>
                          Select a colony marker to remove it
                        </div>
                      )}
                      <div style={{ fontSize: "0.6rem", color: "#3D4A5C", marginTop: 10, textAlign: "center" }}>
                        +{corrections.added} added · -{corrections.removed} removed
                      </div>
                    </div>

                    <button
                      onClick={() => fileRef.current?.click()}
                      style={{
                        width: "100%",
                        padding: "10px",
                        background: SURFACE,
                        color: "#8892A8",
                        border: "1px solid rgba(136,146,168,0.2)",
                        borderRadius: 9,
                        cursor: "pointer",
                        fontWeight: 600,
                        fontSize: "0.78rem",
                        fontFamily: "'Outfit'",
                      }}
                    >
                      New Plate
                    </button>
                    <input ref={fileRef} type="file" accept="image/*" onChange={handleUpload} style={{ display: "none" }} />
                  </div>
                </div>
              )}

              {tab === "labels" && (
                <div style={{ maxWidth: 900, margin: "0 auto" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#E8EAF0", marginBottom: 6 }}>
                    Label Filter Summary
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "#546E7A", marginBottom: 18 }}>
                    These candidates were rejected as likely printed or written plate markings.
                  </div>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    {analysis.labels.slice(0, 60).map((item) => (
                      <div
                        key={item.id}
                        style={{
                          background: CARD,
                          border: "1px solid rgba(255,107,107,0.16)",
                          borderLeft: `4px solid ${RED}`,
                          borderRadius: 10,
                          padding: "14px 16px",
                          flex: "1 1 180px",
                          minWidth: 180,
                        }}
                      >
                        <div style={{ fontFamily: "'JetBrains Mono'", fontSize: "1rem", fontWeight: 700, color: RED }}>
                          {(item.label_score * 100).toFixed(0)}%
                        </div>
                        <div style={{ fontSize: "0.66rem", color: "#8892A8", textTransform: "uppercase", letterSpacing: 1 }}>
                          Label score
                        </div>
                        <div style={{ marginTop: 8, fontSize: "0.68rem", color: "#C0C8D8" }}>
                          x={item.x.toFixed(0)} · y={item.y.toFixed(0)} · r={item.r.toFixed(1)}
                        </div>
                        <div style={{ marginTop: 4, fontSize: "0.68rem", color: "#8892A8" }}>
                          contrast {item.local_contrast.toFixed(0)} · circularity {item.circularity.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {tab === "data" && (
                <div style={{ maxWidth: 920, margin: "0 auto" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#E8EAF0", marginBottom: 12 }}>
                    Colony Detection Results
                  </div>
                  <div style={{ background: CARD, borderRadius: 10, overflow: "hidden", border: "1px solid rgba(212,168,67,0.1)" }}>
                    <div style={{ overflowX: "auto", maxHeight: 420 }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "'JetBrains Mono'", fontSize: "0.7rem" }}>
                        <thead>
                          <tr style={{ background: "rgba(212,168,67,0.08)" }}>
                            {["#", "X", "Y", "Size", "Conf", "Area", "Label Risk"].map((header) => (
                              <th
                                key={header}
                                style={{
                                  padding: "10px 12px",
                                  textAlign: "left",
                                  color: GOLD,
                                  fontWeight: 600,
                                  borderBottom: "1px solid rgba(212,168,67,0.1)",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {filteredColonies.slice(0, 150).map((item, index) => (
                            <tr
                              key={item.id}
                              style={{
                                background: index % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)",
                                cursor: "pointer",
                              }}
                              onClick={() => {
                                setSelectedId(item.id);
                                setTab("detect");
                              }}
                            >
                              <td style={{ padding: "7px 12px", color: "#8892A8" }}>{index + 1}</td>
                              <td style={{ padding: "7px 12px", color: "#C0C8D8" }}>{item.x.toFixed(0)}</td>
                              <td style={{ padding: "7px 12px", color: "#C0C8D8" }}>{item.y.toFixed(0)}</td>
                              <td style={{ padding: "7px 12px" }}>
                                <span
                                  style={{
                                    padding: "2px 8px",
                                    borderRadius: 10,
                                    fontSize: "0.6rem",
                                    background:
                                      item.size === "small"
                                        ? "rgba(79,195,247,0.15)"
                                        : item.size === "medium"
                                          ? "rgba(212,168,67,0.15)"
                                          : "rgba(255,107,107,0.15)",
                                    color:
                                      item.size === "small" ? BLUE : item.size === "medium" ? GOLD : RED,
                                  }}
                                >
                                  {item.size}
                                </span>
                              </td>
                              <td style={{ padding: "7px 12px", color: item.conf > 0.8 ? GREEN : item.conf > 0.6 ? GOLD : RED }}>
                                {(item.conf * 100).toFixed(0)}%
                              </td>
                              <td style={{ padding: "7px 12px", color: "#8892A8" }}>{item.area.toFixed(0)}</td>
                              <td style={{ padding: "7px 12px", color: item.label_score < 0.2 ? GREEN : item.label_score < 0.4 ? GOLD : RED }}>
                                {(item.label_score * 100).toFixed(0)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div style={{ fontSize: "0.65rem", color: "#546E7A", marginTop: 8 }}>
                    Showing {Math.min(filteredColonies.length, 150)} of {filteredColonies.length} colonies above {(threshold * 100).toFixed(0)}% confidence
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
