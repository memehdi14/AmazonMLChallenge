"""Build <team>_submission.zip in the structure required by the challenge:

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, configs/, utils/, README.md, requirements.txt}
└── Documentation_template.md

Usage: python -m src.package_submission --team <team_name>
"""
import argparse
import glob
import os
import zipfile

# teammate placeholders that the final pipeline does not use
EXCLUDE_SRC = {"cross_encoder.py", "export.py"}


def main():
    """Validate that every required file exists, then write the zip."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()
    pkg = "code/business_entity_resolution"
    files = {
        "output/matching_results.tsv": "output/matching_results.tsv",
        "output/candidate_pairs.tsv": "output/candidate_pairs.tsv",
        "Documentation_template.md": "Documentation_template.md",
        f"{pkg}/README.md": "docs/PACKAGE_README.md",
        f"{pkg}/requirements.txt": "requirements.txt",
        f"{pkg}/utils/validate_submission.py": "utils/validate_submission.py",
        f"{pkg}/configs/default.yaml": "configs/default.yaml",
        f"{pkg}/configs/smoke.yaml": "configs/smoke.yaml",
    }
    for f in sorted(glob.glob("src/*.py")):
        if os.path.basename(f) not in EXCLUDE_SRC:
            files[f"{pkg}/src/{os.path.basename(f)}"] = f
    missing = [src for src in files.values() if not os.path.exists(src)]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    path = os.path.join(a.out_dir, f"{a.team}_submission.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in files.items():
            z.write(src, arc)
            print(f"  + {arc}")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
