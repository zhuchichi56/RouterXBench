"""
Script to upload all training loss history JSON files to Weights & Biases (wandb)

Usage:
    python utils/upload_to_wandb.py

This will upload all JSON files from probe_save/loss/ directory.
Only uploads train_loss and val_loss curves, skips already uploaded files.
"""

import json
import argparse
from pathlib import Path
import sys
import os
os.environ["WANDB_API_KEY"] = "79a88980fe13540412ac35e9673ca1ebe5e23380"

# Import wandb at the top - user should set WANDB_API_KEY before running
import wandb


def check_if_run_exists(project_name: str, run_name: str) -> bool:
    """
    Check if a run with the given name already exists in the project

    Args:
        project_name: wandb project name
        run_name: run name to check

    Returns:
        True if run exists, False otherwise
    """
    try:
        api = wandb.Api()
        # Get all runs from the project
        runs = api.runs(f"{api.default_entity}/{project_name}")

        # Check if any run has the same name
        for run in runs:
            if run.name == run_name:
                return True
        return False
    except Exception as e:
        print(f"⚠️  Warning: Could not check for existing runs: {e}")
        return False


def upload_training_history_to_wandb(
    json_file_path: str,
    project_name: str = "probe-training",
    run_name: str = None
):
    """
    Upload training history from JSON file to wandb (train_loss, val_loss, val_acc, val_auroc, lr)

    Args:
        json_file_path: Path to the JSON file containing training history
        project_name: wandb project name
        run_name: Optional custom run name (defaults to filename)
    """
    # Load JSON data
    json_path = Path(json_file_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract data
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    val_accuracies = data.get('val_accuracies', [])
    val_aurocs = data.get('val_aurocs', [])
    learning_rates = data.get('learning_rates', [])

    # Extract metadata
    initial_lr = data.get('initial_lr', 0.0)
    best_val_loss = data.get('best_val_loss', float('inf'))
    best_val_acc = data.get('best_val_acc', 0.0)
    best_val_auroc = data.get('best_val_auroc', 0.0)

    # Prepare run name
    if run_name is None:
        run_name = json_path.stem  # Use filename without extension

    # Check if run already exists
    if check_if_run_exists(project_name, run_name):
        print(f"⏭️  Skipping {run_name} - already uploaded")
        return None

    print(f"📊 Uploading: {run_name}")

    # Initialize wandb run with config
    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            "initial_lr": initial_lr,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_auroc": best_val_auroc,
        },
        reinit=True  # Allow multiple runs in same script
    )

    # Log metrics for each epoch
    for epoch in range(len(train_losses)):
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_losses[epoch],
        }

        # Add validation loss if available
        if epoch < len(val_losses):
            log_dict["val_loss"] = val_losses[epoch]

        # Add validation accuracy if available
        if epoch < len(val_accuracies):
            log_dict["val_acc"] = val_accuracies[epoch]

        # Add validation AUROC if available
        if epoch < len(val_aurocs):
            log_dict["val_auroc"] = val_aurocs[epoch]

        # Add learning rate if available
        if epoch < len(learning_rates):
            log_dict["learning_rate"] = learning_rates[epoch]

        wandb.log(log_dict)

    print(f"✅ Uploaded: {run_name}")

    # Finish the run
    wandb.finish()

    return run.url


def upload_all_json_files(
    loss_dir: str = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/loss",
    project_name: str = "probe-training"
):
    """
    Upload all JSON files from the loss directory to wandb

    Args:
        loss_dir: Directory containing JSON files
        project_name: wandb project name
    """
    loss_path = Path(loss_dir)

    if not loss_path.exists():
        print(f"❌ Directory not found: {loss_dir}")
        return

    # Find all JSON files
    json_files = sorted(loss_path.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {loss_dir}")
        return

    print(f"📁 Found {len(json_files)} JSON files in {loss_dir}")
    print(f"🚀 Starting upload to wandb project: {project_name}")
    print("="*60)

    uploaded_count = 0
    skipped_count = 0
    failed_count = 0

    for json_file in json_files:
        try:
            result = upload_training_history_to_wandb(
                json_file_path=str(json_file),
                project_name=project_name
            )
            if result is None:
                skipped_count += 1
            else:
                uploaded_count += 1
        except Exception as e:
            print(f"❌ Failed to upload {json_file.name}: {e}")
            failed_count += 1

    print("="*60)
    print(f"✅ Upload complete!")
    print(f"   Uploaded: {uploaded_count}")
    print(f"   Skipped (already exists): {skipped_count}")
    print(f"   Failed: {failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload all training loss history JSON files to wandb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python utils/upload_to_wandb.py

Note:
  Before running, set your wandb API key:
    export WANDB_API_KEY="your_api_key_here"

  Or login with:
    wandb login
        """
    )

    parser.add_argument(
        "--project",
        type=str,
        default="probe-training",
        help="wandb project name (default: probe-training)"
    )

    parser.add_argument(
        "--loss-dir",
        type=str,
        default="/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/loss",
        help="Directory containing JSON files (default: /volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/loss)"
    )

    args = parser.parse_args()

    try:
        upload_all_json_files(
            loss_dir=args.loss_dir,
            project_name=args.project
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

