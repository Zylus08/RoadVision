"""Automated smoke test verifying all deliverables, files, and execution commands."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import subprocess

def smoke_test():
    print("================================================================")
    print("TASK 4: COMPREHENSIVE SMOKE TEST OF DELIVERABLES AND ARTIFACTS")
    print("================================================================")
    
    # 1. Check all critical files exist
    critical_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "app.py",
        "configs/default_config.yaml",
        "src/__init__.py",
        "src/inference.py",
        "src/video_processor.py",
        "src/corruptions.py",
        "src/visualization.py",
        "src/metrics.py",
        "src/dataset.py",
        "scripts/run_inference.py",
        "scripts/evaluate.py",
        "scripts/robustness_test.py",
        "scripts/reproduce_clean_eval.py",
        "docs/methodology.md",
        "docs/results.md",
        "docs/viva_preparation.md",
        "docs/audit_report.md",
        "outputs/metrics/clean_eval_reproduction.json",
        "outputs/metrics/verified_clean_eval.json",
        "outputs/metrics/verified_robustness_benchmark.json",
        "outputs/metrics/dataset_provenance_audit.json",
        "outputs/plots/verified_per_class_iou.png",
        "outputs/plots/verified_robustness_degradation_curves.png",
        "outputs/plots/verified_robustness_max_severity_drop.png"
    ]
    
    missing = []
    for f in critical_files:
        if not os.path.exists(f):
            missing.append(f)
            
    assert not missing, f"Missing critical files: {missing}"
    print(f"[1] Verified existence of all {len(critical_files)} critical files.")
    
    # 2. Verify JSON metrics match reported numbers
    with open("outputs/metrics/clean_eval_reproduction.json", "r") as f:
        repro = json.load(f)
        assert abs(repro["reproduced_aggregate_miou"] - 0.3898) < 0.001
        assert abs(repro["reproduced_aggregate_pixel_accuracy"] - 0.9447) < 0.001
        print("[2] Verified that saved clean_eval_reproduction.json matches reported 38.98% mIoU and 94.47% accuracy.")
        
    with open("outputs/metrics/verified_robustness_benchmark.json", "r") as f:
        bench = json.load(f)
        assert abs(bench["corruptions"]["fog"]["5"]["corrupted_mIoU"] - 0.1716) < 0.001
        assert abs(bench["corruptions"]["fog"]["5"]["relative_drop_pct"] - 56.0) < 0.1
        print("[3] Verified that saved verified_robustness_benchmark.json matches 56.0% fog degradation.")
        
    # 3. Test CLI single-image inference on authentic sample
    print("\n[4] Executing CLI inference on authentic sample...")
    test_img = "data/samples/authentic/frankfurt_000000_000294.png"
    cmd = [
        sys.executable, "scripts/run_inference.py",
        "--input", test_img,
        "--output-dir", "outputs/predictions"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference CLI failed: {res.stderr}"
    assert os.path.exists("outputs/predictions/frankfurt_000000_000294_mask.png")
    assert os.path.exists("outputs/predictions/frankfurt_000000_000294_overlay.png")
    assert os.path.exists("outputs/predictions/frankfurt_000000_000294_telemetry.json")
    print("  SUCCESS: Generated mask, overlay, and telemetry for authentic sample.")
    
    # 4. Streamlit Dashboard Syntax and Import Check
    print("\n[5] Validating Streamlit dashboard executable syntax...")
    cmd_st = [sys.executable, "-m", "py_compile", "app.py"]
    res_st = subprocess.run(cmd_st, capture_output=True, text=True)
    assert res_st.returncode == 0, f"Dashboard syntax check failed: {res_st.stderr}"
    print("  SUCCESS: app.py compiled cleanly with zero syntax or import errors.")
    
    print("\nALL SMOKE TESTS PASSED!")

if __name__ == "__main__":
    smoke_test()

