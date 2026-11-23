# mpox-fm-lodo

Code for *Foundation Models vs CNNs vs ViTs for Mpox Detection: A Leave-One-Dataset-Out
Evaluation on Hyperdatasets*. The pipeline runs on a free Kaggle GPU notebook and is split
into stages that can be resumed across sessions.

## Stages

| Stage | Script | Hardware | Output |
|---|---|---|---|
| 1 | `01_build_index.py` | CPU | image index and per-dataset report |
| 2 | `02_audit.py --embed` | GPU | duplicate candidates, blind review sample |
| 3 | `03_groups_splits.py` | CPU | duplicate groups, LODO families, ITA, splits, leakage check |
| 4 | `04_features.py` | GPU | frozen embeddings for 12 models |
| 5 | `05_linear_probe.py` | CPU | linear-probe predictions |
| 6 | `06_zeroshot.py` | GPU | zero-shot predictions (CLIP, BiomedCLIP, MedSigLIP) |
| 7 | `07_finetune.py` | GPU | FT and LP-FT for the best model of each family |
| 8 | `08_efficiency.py` | GPU | parameters, latency, memory |
| 9 | `09_evaluate.py` | CPU | tables, figures, statistics |
| 10 | `10_sensitivity.py` | CPU | results under four duplicate rules |

`run_all.sh` groups them: `setup`, `audit`, `splits`, `features`, `probe`, `finetune`,
`report`, `smoke`.

## Running on Kaggle

1. Notebook settings: GPU T4, internet on.
2. Add these Kaggle datasets as inputs: `shubhamgoel27/dermnet`,
   `dipuiucse/monkeypoxskinimagedataset`, `sachinkumar413/monkeypox-images-dataset`,
   `nafin59/monkeypox-skin-lesion-dataset`, `ahmadnasayrah/data-monkeypox`,
   `maxmelichov/monkeypox-2022-remastered`,
   `joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20`, `piyush19/monkeypox-dataset`,
   `subirbiswas19/skin-disease-dataset`.
3. MedSigLIP is gated: accept its terms on Hugging Face and store a read token as the
   Kaggle secret `HF_TOKEN`. Without it the model is skipped.
4. Clone this repository into `/kaggle/working/code` and run
   `bash run_all.sh setup`. This fetches the two GitHub-only datasets, MPox-Vision
   (Mendeley), MCSI (Zenodo) and the PanDerm weights.
5. Run the remaining stages in order. `index/dataset_report.csv` lists every dataset with the
   number of images found; `unlabelled_dirs` shows folders whose class could not be read.

Outputs are written to `/kaggle/working/mpox_out`. To continue in a new session, attach the
previous notebook's output as an input; `setup` restores it, and finished stages and runs are
skipped. The index and the audit are never rebuilt on resume, so the review sample stays the
same. If the splits change, earlier predictions are moved to `preds_superseded_<hash>`.

## Duplicates

Candidate pairs are those with pHash or dHash distance ≤ 10 or DINOv2 cosine ≥ 0.95. A pair
counts as a duplicate when the files are identical, the pHash distance is ≤ 8 or the cosine is
≥ 0.95. The pre-registered rule (≤ 4 or ≥ 0.98) was replaced after the two-reviewer check
described below (Amendment 2 in `PREREGISTRATION_EXTERNAL.md`); it is still reported in the
sensitivity analysis.

Duplicate pairs are joined into groups. Two sources form one LODO family when more than half
of either one's images are duplicated in the other; MSLD and MSLD v2.0 are always one family.
Training images that share a group with the held-out source are dropped from that fold, and
`assert_no_leakage` stops the run if any group crosses train and test.

### Two-reviewer check

The audit draws 200 candidate pairs (100 accepted and 100 rejected by the rule, each half split
between within- and cross-dataset pairs). Reviewers get `FOR_REVIEWERS.zip`, which holds only
pair numbers, thumbnail sheets and an empty form each (`review_form_R1.csv`,
`review_form_R2.csv`); the rule's answers stay in `audit/review_key.csv`. Each reviewer marks
every pair `d` (duplicate), `n` (near-duplicate) or `x` (distinct). Stage 3 reads the filled
forms from `reviews/` and reports agreement, Cohen's kappa, and the rule's precision and miss
rate in `audit/audit_summary.json`. The completed forms are in `reviews/`.

## Fine-tuning budget

FT runs on three seeds for every LODO fold and on one seed for the pooled folds; LP-FT is run
only for the selected foundation model. The external test uses FT and LP-FT for every selected
model with three seeds, as pre-registered. One ViT-B run takes about 10-30 min on a T4.
`--max-runs` and `--max-hours` stop a session cleanly.

## Pre-registration

`PREREGISTRATION_EXTERNAL.md` fixes the external test before any external image is classified.
Changes are added as dated amendments at the end of the file.

## Tests

`bash tests/run_cpu_tests.sh` runs the CPU stages end to end on synthetic images.

## Licence

Code: MIT. Each dataset keeps its own licence; PanDerm weights are CC BY-NC-ND 4.0.

Mpox2025 was registered as a third external set but is no longer downloadable; see Amendment 3
in `PREREGISTRATION_EXTERNAL.md`.
