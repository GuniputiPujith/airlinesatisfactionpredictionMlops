import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_NAME = "churn-prediction-api:1.0"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

REQUIRED_FILES = [
    DEPLOYMENT_DIR / "model.pkl",
    DEPLOYMENT_DIR / "preprocessor.pkl",
    DEPLOYMENT_DIR / "model_metadata.json",
]

def run_command(command):
    print("\n[RUNNING]", " ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )

def main():
    print("=" * 65)
    print("AIRLINE PASSENGER SATISFACTION - PRODUCTION DEPLOYMENT")
    print("=" * 65)

    # 1. Verify deployment artifacts
    print("\n[INFO] Checking deployment artifacts...")

    for file_path in REQUIRED_FILES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required deployment artifact missing: {file_path}"
            )

        print(f"[SUCCESS] Found: {file_path.name}")

    # 2. Build Docker image
    print("\n[INFO] Building Docker image...")

    run_command([
        "docker",
        "build",
        "-t",
        IMAGE_NAME,
        ".",
    ])

    print("[SUCCESS] Docker image built.")

    # 3. Apply Kubernetes Deployment
    print("\n[INFO] Applying Kubernetes Deployment...")

    run_command([
        "kubectl",
        "apply",
        "-f",
        str(PROJECT_ROOT / "k8s" / "deployment.yaml"),
    ])

    # 4. Apply Kubernetes Service
    print("\n[INFO] Applying Kubernetes Service...")

    run_command([
        "kubectl",
        "apply",
        "-f",
        str(PROJECT_ROOT / "k8s" / "service.yaml"),
    ])

    # 5. Wait for Deployment rollout
    print("\n[INFO] Waiting for Kubernetes rollout...")

    run_command([
        "kubectl",
        "rollout",
        "status",
        "deployment/churn-api",
        "--timeout=180s",
    ])

    print("\n" + "=" * 65)
    print("[SUCCESS] PRODUCTION DEPLOYMENT COMPLETED")
    print("=" * 65)

    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[ERROR] Deployment failed: {error}")
        sys.exit(1)
