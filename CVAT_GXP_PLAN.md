# CVAT + Apricot Colony Counter Integration Plan

Date: 2026-04-26

## Goal

Use CVAT as the manual annotation system for Apricot Colony Counter so we can:

1. build a real benchmark set,
2. stop tuning blindly on one or two plates,
3. collect per-image user guidance during upload,
4. store that guidance separately from gold-standard annotations,
5. improve both offline training and live per-plate inference.

## Why CVAT

CVAT is a strong fit because it supports the annotation shapes we actually need:

- points
- ellipses
- polygons
- masks

Its native image export format supports those shapes directly, and CVAT also exposes a REST API / SDK for automation.

Useful docs:

- CVAT overview: https://docs.cvat.ai/docs/getting_started/overview/
- CVAT API: https://docs.cvat.ai/docs/api_sdk/api/
- CVAT native image format: https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
- COCO in CVAT: https://docs.cvat.ai/docs/dataset_management/formats/format-coco/

## Recommended Annotation Strategy

Do not start with pixel-perfect masks for every colony.

That is too slow and unnecessary for the current stage.

Instead, define 5 annotation types:

1. `dish`
   - shape: ellipse
   - purpose: plate localization benchmark

2. `colony_point`
   - shape: point
   - purpose: counting and detection evaluation for small dot-like colonies

3. `colony_ellipse`
   - shape: ellipse
   - purpose: counting and size-aware evaluation for larger diffuse/blob-like colonies

4. `label_region`
   - shape: polygon or mask
   - purpose: directly supervise "label vs not label"

5. `ignore_region`
   - shape: polygon
   - purpose: glare, scratches, rim junk, ambiguous artifacts

### Minimal viable annotation

For each plate:

- 1 dish ellipse
- colony center points for small dot-like colonies
- colony ellipses for large diffuse/blob-like colonies
- 1+ label polygons over printed text / markings
- optional ignore polygons

This is fast enough to scale and useful enough to drive the pipeline.

## Two Different Data Streams

We should separate annotation data into two categories.

### A. Gold annotations

These come from CVAT and are the trusted benchmark / training data.

Examples:

- full colony annotations with points for small colonies and ellipses for large diffuse colonies
- reviewed label regions
- ignore regions

Use for:

- benchmark evaluation
- threshold tuning
- later supervised learning

### B. Per-image user hints

These come from the Apricot upload flow when a lab user clicks a few obvious colonies.

Examples:

- 3 to 8 positive colony clicks
- optionally 1 to 3 "not colony" clicks
- optionally a quick label scribble / polygon later

Use for:

- image-specific adaptation
- active learning queue
- future annotation task generation

Important:

These should **not** automatically become gold labels.
They are weak supervision until reviewed or promoted.

## Proposed Apricot Upload UX

When a user uploads a plate:

1. Apricot runs the normal automatic pipeline.
2. Before showing the final count, ask for a tiny calibration step:
   - "Click 3 to 5 obvious colonies."
3. Optional second step:
   - "Click 1 to 3 obvious non-colony dots or label marks."
4. Then rerun or rescore the candidates using those clicks as per-image priors.

This is still much less work than manual colony counting, but it gives the model plate-specific context.

## How To Use The User Clicks

The user clicks can help in two ways.

### 1. Live image-specific adaptation

From the positive colony clicks, estimate:

- local colony LAB color distribution
- typical radius range
- typical local contrast

Then use those values to adjust candidate ranking for that plate.

Examples:

- boost candidates close to clicked colony color
- suppress candidates far from clicked colony color
- tighten the label filter if clicked colonies are clearly warm/golden

This is not full retraining. It is per-image calibration.

### 2. Offline dataset growth

Store the user clicks along with:

- image id
- timestamp
- model version
- final reviewed count if available

Later, use them to:

- choose which images to send to CVAT
- pre-seed annotation tasks
- compare user hints vs final gold annotation

## Recommended Storage Model

Keep these separate:

### `gold_annotations`

Source of truth from CVAT export.

Fields:

- image_id
- source = `cvat`
- dish
- colony points / ellipses
- label regions
- ignore regions
- annotator
- review status
- created_at

### `user_hints`

Weak supervision from the Apricot upload flow.

Fields:

- image_id
- source = `gxp_user`
- positive_clicks
- negative_clicks
- optional label hints
- model_version
- user_id or lab_id
- created_at

### `model_runs`

Inference metadata.

Fields:

- image_id
- model_version
- candidate_count
- colony_count
- label_count
- confidence threshold
- inference timing
- deployed commit hash

## CVAT Integration Levels

### Level 1: manual / low-friction integration

Best starting point.

Workflow:

1. Export candidate images from Apricot.
2. Upload them into a CVAT project.
3. Annotate manually in CVAT.
4. Export annotations in CVAT native format.
5. Convert those exports into Apricot benchmark JSON with a repo script.

This is enough to start building a real benchmark set immediately.

### Level 2: API-assisted integration

After the benchmark workflow works, automate it with the CVAT API.

Potential flow:

1. "Send to CVAT" button in Apricot for hard cases.
2. Apricot creates a CVAT task via API.
3. Task includes:
   - original image
   - optional model predictions as pre-annotations
4. Annotator reviews in CVAT.
5. Nightly import job pulls completed annotations back into Apricot.

### Level 3: review loop

Later:

- hard images from the live app automatically enter an annotation queue
- reviewed CVAT annotations feed benchmark and future training

## Recommended Export Format

For this project:

### Primary archive format

Use CVAT native image export first.

Reason:

- supports points, ellipses, polygons, masks directly
- less lossy for this mixed annotation setup
- good as a source-of-truth backup

### Training / interoperability format later

Export COCO only if a downstream model pipeline specifically needs it.

Reason:

- COCO is widely supported
- but mixed shape semantics can be less natural than CVAT native for this use case

## Immediate Next Steps

1. Create one CVAT project for Apricot colony-counter plates.
2. Define labels:
   - `dish`
   - `colony_point`
   - `colony_ellipse`
   - `label_region`
   - `ignore_region`
3. Annotate 20 to 50 plates across:
   - sparse
   - dense
   - bright
   - dark
   - different label placements
4. Add a repo script to convert CVAT export into Apricot evaluation JSON.
5. Add a lightweight Apricot UI step for:
   - 3 to 5 positive colony clicks
   - optional 1 to 3 negative clicks

## Important Guardrail

Do not tune the model against one image like `1399.jpg` until it "looks right."

Instead:

- use `1399.jpg` as one failure case,
- build a benchmark set in CVAT,
- measure changes across many plates,
- keep user hints separate from gold labels.

That is how we avoid overtraining on a handful of images while still using human input intelligently.
