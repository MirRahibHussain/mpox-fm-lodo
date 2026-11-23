"""Stage 2: duplicate audit.

CPU part: pHash/dHash candidates. GPU part (--embed): DINOv2 embeddings and
cosine candidates. Writes candidates_for_review.csv and contact sheets.
Two annotators then fill reviewer_1 / reviewer_2 (duplicate, near_duplicate,
distinct) and, after discussion, an optional `final` column. Save the edited
file as candidates_reviewed.csv in the same folder and run stage 3.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from mpoxlodo import dedup, io

REVIEW_INSTRUCTIONS = """Duplicate review - {n} image pairs

Two reviewers work INDEPENDENTLY: reviewer 1 fills review_form_R1.csv, reviewer 2 fills
review_form_R2.csv. Do not look at or discuss each other's answers until both are done.

For each pair number, open the sheet in sheets/ that contains it (A = left, B = right) and write
in the 'decision' column one of:
  duplicate       the same photograph (possibly resized, recompressed or with a watermark)
  near_duplicate  the same photograph after a crop, flip, rotation or colour/contrast change
  distinct        different photographs (even if they show the same patient or lesion type)
Short forms are accepted: d / n / x.

Judge only whether the two images come from the same photograph. Ignore the disease.
When both forms are filled, upload both CSV files together as a private Kaggle dataset named
'mpox-review' and add it to the notebook as an input. Disagreements are discussed afterwards;
the original answers are kept for Cohen's kappa.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="also run DINOv2 embedding candidates (GPU)")
    ap.add_argument("--phash-threshold", type=int, default=dedup.PHASH_THRESHOLD)
    ap.add_argument("--cos-threshold", type=float, default=dedup.EMBED_THRESHOLD)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--full-sheets", action="store_true",
                    help="sheets for every unresolved pair (default: only the validation sample)")
    ap.add_argument("--sample", type=int, default=200, help="pairs in the two-reviewer validation sample")
    ap.add_argument("--force", action="store_true", help="redo the audit even if it exists")
    a = ap.parse_args()
    if os.path.exists(io.p("audit", "review_key.csv")) and not a.force:
        print("audit exists (resumed run): kept, so the reviewers' sample stays identical. Use --force to redo.")
        sys.exit(0)

    index = io.load_index()
    hashed = dedup.compute_hashes(index)
    hashed[["image_id", "phash", "dhash"]].to_csv(io.p("audit", "hashes.csv"), index=False)
    hc = dedup.hash_candidates(hashed, a.phash_threshold)
    print(f"hash candidates: {len(hc)}")

    ec = None
    if a.embed:
        from mpoxlodo import features
        fn = features.extract(index, "dinov2_b14", io.p("features"))
        ids, emb = features.load(io.p("features"), "dinov2_b14")
        ec = dedup.embedding_candidates(list(ids), emb, a.cos_threshold)
        print(f"embedding candidates: {len(ec)}")

    cands = dedup.merge_candidates(hc, ec, index)
    out = io.p("audit", "candidates_for_review.csv")
    cands.to_csv(out, index=False)
    need = int((~cands["auto_duplicate"]).sum())
    print(f"{len(cands)} candidate pairs; {int(cands['auto_duplicate'].sum())} auto-accepted; {need} need review")
    print(f"cross-dataset pairs: {int(cands['cross_dataset'].sum())}")
    smp = dedup.review_sample(cands, a.sample)
    key, form = dedup.review_forms(smp)
    key.to_csv(io.p("audit", "review_key.csv"), index=False)   # hidden from reviewers
    rdir = io.p("audit", "for_reviewers")
    os.makedirs(rdir, exist_ok=True)
    form.to_csv(os.path.join(rdir, "review_form_R1.csv"), index=False)
    form.to_csv(os.path.join(rdir, "review_form_R2.csv"), index=False)
    with open(os.path.join(rdir, "INSTRUCTIONS.txt"), "w") as f:
        f.write(REVIEW_INSTRUCTIONS.format(n=len(form)))
    print(f"validation sample: {len(smp)} pairs -> audit/for_reviewers/ "
          f"(rule says duplicate for {int(smp['rule_duplicate'].sum())}; key kept in audit/review_key.csv)")
    if not a.no_sheets:
        files = dedup.write_review_sheet(smp, index, os.path.join(rdir, "sheets"))
        print(f"wrote {len(files)} sample review sheets")
        if a.full_sheets and need:
            files = dedup.write_review_sheet(cands, index, io.p("audit", "review_sheets"))
            print(f"wrote {len(files)} full review sheets")
    # blind package for the reviewers (no key, no scores, no dataset names)
    import shutil
    dest = os.path.join(os.path.dirname(os.path.abspath(io.p())), "FOR_REVIEWERS")
    shutil.make_archive(dest, "zip", rdir)
    print(f"reviewer package: {dest}.zip")
