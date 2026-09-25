"""
Amazon ML Challenge 2026 - Master Execution Script
Runs the complete Phase 1 -> Phase 2 -> Phase 3 pipeline and generates
the final leaderboard submission package.

Usage (in Anaconda PowerShell):
    python run_pipeline.py
"""

import os
import sys
import subprocess

# Ensure project src is in Python path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(workspace_dir, "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

# 1. Check & install dependencies if needed
REQUIRED_PACKAGES = ["lightgbm", "rapidfuzz", "jellyfish", "anyascii", "scikit-learn", "pandas", "numpy", "joblib"]

print("="*70)
print("=== AMAZON ML CHALLENGE 2026: END-TO-END EXECUTION ===")
print("="*70, flush=True)

missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Installing missing packages: {missing}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    print("Dependencies installed successfully.\n")
else:
    print("All required packages are installed and ready.\n")

# 2. Paths
dataset_dir = os.path.join(workspace_dir, "Dataset", "student_resource", "dataset")
output_dir = os.path.join(workspace_dir, "output")
artifacts_dir = os.path.join(workspace_dir, "code", "business_entity_resolution", "artifacts")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

# 3. Import pipeline modules
from pipeline import run_validation_pipeline
from generate_submission import run_test_submission

# Step 1: Train & Calibrate Matching Model on Validation Set
print("STEP 1: Training LightGBM Model & Optimizing Threshold for Macro F_0.5...")
model_path = os.path.join(artifacts_dir, "lgbm_model.pkl")
config_path = os.path.join(artifacts_dir, "config.json")

# Train and sweep threshold if not already cached
if not (os.path.exists(model_path) and os.path.exists(config_path)):
    run_validation_pipeline(data_dir=dataset_dir, top_k=50, num_s1=2000)
else:
    print(f"Using cached trained model from {artifacts_dir}")

# Step 2: Generate Final Test Submission Files (matching_results.tsv & candidate_pairs.tsv)
print("\nSTEP 2: Generating Test Candidates & Final Scored Matches (Including France)...")
run_test_submission(
    model_path=model_path,
    config_path=config_path,
    data_dir=dataset_dir,
    output_dir=output_dir,
    top_k=50
)

# Step 3: Run Official Submission Validator
print("\nSTEP 3: Validating Output Format with Official Validator...")
validator_script = os.path.join(workspace_dir, "Dataset", "student_resource", "utils", "validate_submission.py")
matching_file = os.path.join(output_dir, "matching_results.tsv")
candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
test_directory = os.path.join(dataset_dir, "test")

val_cmd = [
    sys.executable, validator_script,
    "--matching", matching_file,
    "--candidate", candidate_file,
    "--test-dir", test_directory
]
res = subprocess.run(val_cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print(res.stderr)

if res.returncode == 0:
    print("\n" + "="*70)
    print("SUCCESS: Submission files are validated and ready to upload!")
    print(f"Leaderboard file to upload: {matching_file}")
    print("="*70)
else:
    print(f"Validation finished with code: {res.returncode}")
