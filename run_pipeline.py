#!/usr/bin/env python
"""
Run the ML pipeline components sequentially without Kubeflow.
This is a simpler alternative to Kubeflow Pipelines for development/testing.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_pipeline():
    """Run all pipeline components in sequence."""
    
    print("=" * 60)
    print("ML Pipeline Execution (Direct Mode)")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # Component 1-4: Run the training script
    # (includes load data, train, evaluate, and register)
    print("\nRunning training pipeline...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    result = subprocess.run(
        [sys.executable, "src/training/train.py"],
        cwd=str(project_root),
        env=env
    )
    
    if result.returncode != 0:
        print("\n✗ Pipeline execution failed")
        return False
    
    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully")
    print("=" * 60)
    print("\nTo verify the model was registered:")
    print("  kubectl port-forward -n mlops svc/mlflow 5000:5000")
    print("  curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=iris-random-forest")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
