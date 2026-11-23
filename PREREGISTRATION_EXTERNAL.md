# Pre-registration: external post-review test

Commit this file to the public repository **before any external image
(MPox-Vision, Mpox2025, MCSI) is used for inference**. Record the commit hash
and time in the manuscript (Section IV-F) and in Supplementary File S3. Do not
edit it afterwards; add dated amendments at the end instead.

- Date and time of commit (UTC): 2026-11-23 (exact time = timestamp of the Git commit that adds this file)
- Results already known when this plan was written: **no model results**. Only the image index, the duplicate-candidate audit and the cross-dataset overlap matrix of the development datasets had been computed. No pooled, LODO or external performance figure had been produced. No external image had been passed through any classifier.

## 1. Data
- External datasets: MPox-Vision (Mendeley, v2, CC BY 4.0), Mpox2025 (authors' GitHub link, research use), MCSI (Zenodo 8360076, CC BY 4.0).
- Duplicate candidates: pHash or dHash Hamming distance <= 10, or DINOv2-B/14 cosine >= 0.95.
- Duplicate decision: automatic rule (byte-identical, identical perceptual hash, pHash Hamming <= 4, or DINOv2 cosine >= 0.98), validated by two independent human reviewers on a stratified random sample of 400 candidate pairs (blinded to the rule; Cohen's kappa, rule precision and miss rate reported). Pairs are joined into duplicate groups by union-find.
- Duplicate removal: every external image whose duplicate group contains a development image is removed from the primary test. Results with these images kept are reported separately.
- No external image is used for training, early stopping, hyperparameters, thresholds or prompt design.

## 2. Models and training
- Linear probe: all 12 models, trained on the full deduplicated development pool. MedSigLIP-448 (gated) and PanDerm (weights from the authors' link) are included only if their weights are accessible; a model that cannot be loaded is reported as missing, not replaced.
- Fine-tuning and LP-then-FT: the best model of each family, selected by pooled linear-probe balanced accuracy on the development pool (file results/ft_model_selection.csv), 3 seeds.
- Decision threshold: 0.5 for the primary metrics; the 90%-sensitivity threshold is chosen on the development validation split.

## 3. Outcomes
- Primary: balanced accuracy on each external dataset (binary: Mpox vs non-Mpox), mean over seeds.
- Secondary: sensitivity, specificity, AUROC, specificity at 90% sensitivity, share of each non-Mpox class predicted as Mpox, difference between results with and without duplicates.

## 4. Hypotheses
- H1: Mean external balanced accuracy is lower than mean pooled balanced accuracy for every model family.
- H2: The best foundation model has higher external balanced accuracy than the best CNN.
- H3: Including duplicates of development images raises external balanced accuracy.
- H4 (no direction predicted): the model ranking on external data agrees with the LODO ranking (Spearman rho).

## 5. Statistics
- 95% CIs by 2,000 stratified bootstrap resamples per dataset.
- H2: paired comparison over (dataset, seed) pairs with an exact Wilcoxon signed-rank test; Holm correction across the secondary family comparisons.
- Datasets with fewer than 20 Mpox images after duplicate removal report specificity-based outcomes only.

## Amendments
- **Amendment 1 (2026-11-23, before any external inference and before any model result):**
  - Validation sample: 200 candidate pairs instead of 400 (100 the rule accepts, 100 it rejects, each half split
    between within- and cross-dataset pairs). The two reviewers fill separate forms that show only a pair number and
    two thumbnails, so they are blind to the rule, to dataset names and to each other. Reported: Cohen's kappa
    (three categories and duplicate-vs-distinct), rule precision, miss rate among rejected candidates.
  - Added sensitivity analysis (secondary, not a hypothesis test): duplicate groups, LODO families and pooled
    grouped folds are rebuilt under a strict rule (pHash <= 2 or cosine >= 0.99), the main rule (<= 4, >= 0.98)
    and a loose rule (<= 8, >= 0.95); a linear probe with fixed C = 0.1 is trained for every model. Reported per
    rule: pairs, cross-source groups, families, mean pooled and LODO balanced accuracy per model family, the gap,
    and Spearman rho of the LODO ranking against the main rule. External data are not used.
- **Amendment 2 (2026-11-23, before any external inference):**
  - Both reviewer forms were committed (784d452, 1be6786) before being joined to the hidden key.
    Agreement 191/200 (kappa 0.92; duplicate-vs-distinct 0.86). Against the consensus, the registered rule
    (pHash <= 4 or cosine >= 0.98) had precision 1.00 but missed 72 of 94 rejected candidates that both
    reviewers called duplicates; weighted by stratum size its recall within the candidate set is 0.64.
    The rule pHash <= 8 or cosine >= 0.95 (the "loose" rule of Amendment 1) has precision 0.95 and recall 0.99.
  - Change: the main duplicate rule becomes pHash <= 8 or cosine >= 0.95 (or byte-identical / identical hash).
    The registered rule is reported as a sensitivity analysis together with a strict rule (<= 2, >= 0.99) and
    the loosest rule (every candidate: pHash <= 10 or cosine >= 0.95).
  - Disclosure: when this change was made, provisional development-set linear-probe results under the strict,
    registered and loose rules (Amendment 1 sensitivity run) had been computed; they showed similar LODO
    balanced accuracy across rules (model-family means within 0.01) and LODO rankings that agreed (Spearman 0.95-0.98).
    The change is motivated by the reviewers' validation, not by those results. No external image had been
    passed through a classifier.
  - Fine-tuning on development folds (not part of this external plan): 3 seeds on LODO folds, 1 seed on pooled
    folds, LP-FT only for the selected foundation model, because of the GPU budget. External fine-tuning stays as
    registered (FT and LP-FT for each selected model, 3 seeds).
- **Note (2026-11-23, before any external inference):** at the authors' request the repository history was
  squashed into a single commit. The commit hashes cited above (8620e24 for the original registration,
  784d452 and 1be6786 for the two reviewer forms) no longer exist on GitHub. The text of the registration and of
  both amendments is unchanged, and no external image had been passed through a classifier at the time of the
  new commit.
- **Amendment 3 (2026-11-23, before any external inference):** Mpox2025 is dropped from the external test. Its only
  download link (Google Drive file 1P3ZZ01TSJS4v9uhBy0USzfbMXWbO8QIh, given in the authors' repository
  github.com/AAM-Mustahid/MicroMPoxNet) returns "file does not exist", and no other copy could be found. No Mpox2025
  image was ever downloaded or classified. The external test uses MPox-Vision and MCSI; all hypotheses, metrics and
  tests stay as registered and are computed over these two sets.

## Note on development-set design (fixed before this commit)
- LODO families: two development sources are held out together when more than 50% of either one's images fall in duplicate groups shared with the other (MSLD and MSLD v2.0 are always one family). Smaller overlaps are removed from the training side of each fold. This threshold was set after inspecting the overlap matrix and before any model result was produced, because a 5% threshold merged 7 of 11 sources into one family.
