# Apricot Colony YOLOv8n Model Cards

## Model Names

`apricot_clean_dot_counter_v1.pt`

Clean-dot counter specialist for separated round colonies.

`apricot_merged_colony_counter_v1.pt`

Merged-colony counter specialist for touching or partially merged round colonies, including two-lobed snowman-like clusters.

## Training Data

The clean-dot workflow uses Apricot's synthetic generator in `src/apricot/synthetic.py`, producing YOLO-format plate images, `dataset.yaml`, and `manifest.json` with `schema: clean_dots`.

The merged-colony workflow must use a separate YOLO dataset whose manifests declare `schema: merged_snowman`. It trains with `scripts/train_yolo.py --schema merged_snowman` and writes `models/apricot_merged_colony_counter_v1.pt`.

The clean-dot workflow keeps a guard that refuses merged-snowman or streak-line data unless the operator explicitly passes `--include-schema <schema>` for a deliberate mixed-schema experiment.

## Intended Use

Preliminary yeast colony counting from plate images in the Apricot web app. The model is intended to support fast screening and workflow prototyping, not final validated lab reporting.

## Target Species And Media

Target organisms for v0:

- `Saccharomyces cerevisiae`
- `Pichia pastoris` / `Komagataella phaffii`

Current common media/context:

- YPD yeast plates
- LSLB / `E. coli` antibiotic-selection contexts

## Known Tunable Parameter

The primary runtime tuning parameter is the confidence threshold. Lower values increase sensitivity and false positives; higher values reduce false positives and may miss faint or small colonies.

## Limitations

These models still need validation on real Dr. Ford plates. They have not yet been validated for all yeast species, fluorescent strains, unusual media, dense overlapping colonies, heavy condensation, strong glare, unusual plate labels, or plates with colony morphology far outside their declared specialist schema.

## Operational Notes

The web app must not train these models at startup or during request handling. Train offline, then place clean-dot weights at `models/apricot_clean_dot_counter_v1.pt` and merged-colony weights at `models/apricot_merged_colony_counter_v1.pt`. Use `APRICOT_MODEL_PATH` for the clean-dot specialist and `APRICOT_MERGED_MODEL_PATH` for the merged-colony specialist.
