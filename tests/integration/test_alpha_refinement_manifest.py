"""New restitution planes reuse each coordinate's alpha=0.5 attempt stream."""
import csv
import subprocess
import sys


def test_new_planes_share_the_alpha_half_seed(tmp_path):
    base = tmp_path / "grid.csv"
    rows = []
    for alpha in (0.5, 0.8, 1.0):
        for theta, ar, seed in ((1.0, 2.0, 11), (0.2, 3.0, 22)):
            rows.append({"alpha": alpha, "theta": theta, "aspect_ratio": ar, "ensemble_id": 0,
                         "seed": seed + int(alpha * 10), "output_directory": "x"})
    with base.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    out = tmp_path / "new.csv"
    subprocess.run([sys.executable, "hpc/make_alpha_refinement_manifest.py", "--base-grid", str(base),
                    "--alphas", "0.65,0.8", "--output", str(out)], check=True)
    new = list(csv.DictReader(out.open()))
    assert len(new) == 2                                   # 0.8 already exists
    for row in new:
        assert float(row["alpha"]) == 0.65
        expected = 11 + 5 if float(row["aspect_ratio"]) == 2.0 else 22 + 5
        assert int(row["seed"]) == expected                # the alpha = 0.5 seed
        assert row["nsamples"] == "200000"
        assert len(row) == 13                              # run_ctc_row.sh closure format
    subprocess.run([sys.executable, "hpc/make_alpha_refinement_manifest.py", "--base-grid", str(base),
                    "--alphas", "0.65", "--aspect-ratios", "3", "--output", str(out)], check=True)
    assert len(list(csv.DictReader(out.open()))) == 1
