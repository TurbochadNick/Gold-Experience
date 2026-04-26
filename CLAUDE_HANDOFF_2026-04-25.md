# Gold Experience Handoff for Claude

Date: 2026-04-25
Repo: `TurbochadNick/Gold-Experience`
Live URL: `https://gold-experience.onrender.com`

## 1. Current Stack

- Backend: Python + OpenCV
- Web server: stdlib `ThreadingHTTPServer`
- Frontend: no-build static app (`web/index.html`, `web/app.js`, `web/styles.css`)
- Hosting: Render web service
- Current plan: Starter ($7/month)

## 2. Tonight's Code Updates

These commits are already on `origin/main` and deployed to Render.

1. `64885bf` — `Add agar baseline normalization and relative scoring`
   - Added `src/gold_experience/illumination.py`
   - Pipeline now does:
     - CLAHE on L channel
     - per-image agar baseline estimate
     - LAB normalization to canonical agar
   - `candidate_detection.py` now accepts `baseline`
   - `colony_scoring.py` now accepts `agar_baseline`
   - `features.py` now uses relative warmth/darkness/contrast
   - `frontend_payload.py` and `models.py` now expose `agar_baseline`

2. `2bb8e02` — `Downscale large uploads before analysis`
   - `api_server.py` now downsizes uploads before analysis
   - Longest side capped at 1600 px
   - Uses `cv2.INTER_AREA`
   - Goal: cut RAM use and latency on large 4K plate uploads

3. `01e811c` — `Harden analyze response handling and log timings`
   - `web/app.js` now handles empty/invalid JSON responses safely
   - `api_server.py` now logs timings for:
     - upload parse
     - decode
     - resize
     - pipeline
     - payload
     - total

## 3. Render / Runtime Status

- Earlier failures were consistent with Render memory pressure on the free tier.
- After moving to Starter and adding server-side downscaling, the app is operational again.
- Render deploy history shows the three commits above went live successfully.

## 4. Current Problem

The app is operational, but colony detection quality is poor on dense plates.

The strongest observed failure case is:

- `AGAR_representative/1399.jpg`

The live UI screenshot showed:

- `Visible Colonies: 1`
- `Labels Filtered: 1181`
- `Candidates: 1389`

Important nuance:

- The backend is not actually predicting only 1 colony.
- The frontend stat is **thresholded visible colonies**, not raw backend colony count.
- On the same image through the current server path, local analysis produced:
  - `candidates = 1389`
  - `labels = 1181`
  - `colonies = 111`

So there are two problems:

1. The backend is still generating far too many false blob candidates.
2. The frontend can hide most predicted colonies if the confidence slider is high, which makes debugging more confusing.

## 5. Concrete Diagnostics from the Current Code

### A. Candidate generation is too loose

Main hotspot:

- `src/gold_experience/candidate_detection.py`

Why:

1. The detector builds proposals from three generous feature maps:
   - `dark_small`
   - `large_feature`
   - `dark_small` again as a direct proposal source

2. It fuses dark and warm maps with `cv2.max(...)`, which is permissive.

3. `_region_from_proposal(...)` accepts weak local segments around the blob seed and falls back to drawing a disk if no component is found.

4. The bottom-of-function filters are still relatively weak after normalization.

Relevant lines:

- proposal features: `candidate_detection.py:248-269`
- proposal generation: `candidate_detection.py:280-310`
- local region growth / fallback disk logic: `candidate_detection.py:61-118`
- weak candidate acceptance filters: `candidate_detection.py:312-338`

Observed effect on `1399.jpg`:

- Many tiny or weak local maxima become candidates, even where there is no real colony.
- Rim artifacts and plate texture still become proposals.

### B. Label filter is over-classifying dense colony clouds as labels

Main hotspot:

- `src/gold_experience/label_filter.py`

Why:

1. The label logic mostly uses:
   - small size
   - nearby neighbors
   - mild alignment
   - similar radii
   - component size >= 5

2. That is good for dot-matrix text, but dense small colonies also satisfy those heuristics.

3. The component acceptance rule is too permissive:
   - `mean_label_score >= 0.38`
   - `mean_warmth <= 0.40`
   - `mean_radius <= 7.5`
   - `radius_cv <= 0.35`
   - `mean_alignment >= 0.2`

4. The per-candidate fallback rule is also too permissive:
   - `label_score >= 0.40`
   - and either one aligned neighbor or two nearby neighbors

Relevant lines:

- neighbor graph building: `label_filter.py:20-45`
- per-candidate label score: `label_filter.py:47-74`
- component label decision: `label_filter.py:80-114`
- fallback single-candidate label rule: `label_filter.py:116-125`

Observed effect on `1399.jpg`:

- The model marks huge swaths of real small colonies as labels.
- That is why the red overlay dominates the plate.

### C. Colony scorer still admits tiny specks too easily

Main hotspot:

- `src/gold_experience/colony_scoring.py`

Why:

1. `size_score` is binary rather than strongly suppressive below a safe radius.

2. The final threshold `colony_score >= 0.45` is still low enough that tiny high-contrast specks can survive.

3. Several actual colony predictions on `1399.jpg` are very small:
   - area ~25 to 70 px²
   - radius ~2.8 to 4.7 px

These are not reliable colony candidates on a dense plate like this.

Relevant lines:

- scoring setup: `colony_scoring.py:31-39`
- final score / threshold: `colony_scoring.py:41-57`

Observed effect:

- Even after the label filter removes a lot of candidates, the remaining scorer still keeps some tiny junk.

### D. Frontend stats are misleading for debugging

Main hotspot:

- `web/app.js`

Why:

1. The UI’s “Visible Colonies” stat uses `getVisibleColonies()`, which applies the confidence slider threshold.

2. That means the main count card is **not** raw backend colony count.

3. On a dense failure case, this can make the system look like it found 1 colony when the backend actually found 111.

Relevant lines:

- threshold state: `app.js:1-15`
- visible colony filter: `app.js:98-100`
- stats cards: `app.js:176-198`

Observed effect:

- Debugging is harder because the UI conflates “predicted colonies” with “currently visible above threshold”.

## 6. Local Reproduction on `1399.jpg`

Current server path:

- Original image: `3790x4000`
- Downscaled analysis image: `1516x1600`
- Scale: `0.4`

Current output:

- candidates: `1389`
- labels: `1181`
- colonies: `111`

Class averages:

- Labels:
  - avg area: `263.8`
  - avg radius: `7.02`
  - avg contrast: `164.1`
  - avg label score: `0.645`

- Colonies:
  - avg area: `866.0`
  - avg radius: `11.64`
  - avg contrast: `132.5`
  - avg label score: `0.370`
  - avg colony score: `0.597`

- Rejected:
  - avg area: `190.2`
  - avg radius: `4.70`
  - avg contrast: `119.7`
  - avg label score: `0.421`
  - avg colony score: `0.294`

Interpretation:

- The model is not merely “missing a few labels.”
- It is structurally over-generating candidate blobs.
- Then it is structurally confusing dense small colony neighborhoods with dot-matrix labels.

## 7. Things That Are Not the Main Problem

These are not the main cause of the current hallucination behavior:

- Render tier
- Docker
- JSON parsing
- static frontend architecture
- auth / database / product infra

Those mattered for runtime stability, but not for the current CV failure mode.

## 8. Recommended Next Fix Order

### Priority 1 — Tighten candidate generation

Goal:

- Reduce 1389 candidates to something much closer to the real plausible object count.

Recommended changes:

1. Add stronger rim suppression.
2. Increase minimum area / minimum radius for colony proposals.
3. Require stronger local support in `_region_from_proposal(...)`.
4. Stop using the direct fallback disk so eagerly when component extraction fails.
5. Consider rejecting candidates whose mask occupancy inside the bounding box is too low.

### Priority 2 — Make label filtering actually text-like

Goal:

- Prevent dense colony clouds from being interpreted as dot-matrix labels.

Recommended changes:

1. Require true line-like structure for label components.
2. Add anisotropy / PCA axis ratio / component elongation checks.
3. Drop or heavily tighten the fallback rule at `label_filter.py:121-123`.
4. Require stronger multi-point collinearity before assigning label IDs.

### Priority 3 — Make colony scoring more hostile to tiny specks

Goal:

- Stop tiny high-contrast noise from surviving the colony gate.

Recommended changes:

1. Replace binary `size_score` with a steeper size prior.
2. Raise the minimum effective radius for colony acceptance on dense plates.
3. Increase final threshold from `0.45` after retuning.
4. Penalize small/high-contrast/low-area blobs more aggressively.

### Priority 4 — Improve UI observability

Goal:

- Make it obvious what the backend actually predicted.

Recommended changes:

1. Show both:
   - raw predicted colonies
   - visible colonies after threshold

2. Default `showLabels` off for normal review.
3. Keep labels available in debug mode.

## 9. Proposed Next Engineering Pass

If continuing immediately, the next pass should be:

1. Patch `candidate_detection.py`
   - stronger min radius / area
   - stronger rim suppression
   - stricter proposal-to-mask conversion

2. Patch `label_filter.py`
   - require actual line-like text structure
   - tighten or remove fallback label assignment

3. Patch `web/app.js`
   - separate raw colony count from thresholded visible count

## 10. Bottom-Line Diagnosis

The current main failure is **not** that the deployed app is unstable.

The current main failure is:

- the detector produces too many candidates
- the label filter is over-generalizing from “small nearby dots” to “label”
- the colony scorer still lets tiny specks survive
- the frontend count card is confusing because it shows thresholded visible colonies, not raw predicted colonies

That is why dense plates like `1399.jpg` look so wrong right now.

## 11. Follow-Up Experiment: Claude Warmth-Veto Patch

I tested a follow-up patch inspired by Claude's idea:

- stronger warmth veto in `label_filter.py`
- much tighter fallback label rule
- lower allowed mean warmth for label components

### Result on `AGAR_representative/1399.jpg`

Before patch:

- candidates: `1389`
- labels: `1181`
- colonies: `111`

After patch:

- candidates: `1389`
- labels: `77`
- colonies: `989`

### Interpretation

This is a useful result even though it is not deployable.

What it proves:

1. Claude was right that the label filter was too aggressive.
2. Warmth is a strong discriminative cue on this plate.
3. The previous label filter was indeed swallowing large numbers of real colonies.

What it also proves:

1. The candidate generator is still wildly too loose.
2. Once the label gate stops suppressing candidates, the colony scorer is far too permissive.
3. So the system's next bottleneck is still candidate generation and colony scoring, not just label filtering.

### Practical conclusion

The warmth-veto change should not be shipped by itself.

It is a strong clue, not a final fix:

- keep the idea that warm golden colonies should resist label classification
- but only after candidate generation and colony scoring are tightened
