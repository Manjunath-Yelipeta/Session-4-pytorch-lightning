import os
from pathlib import Path
import glob
import logging
from datetime import datetime

log = logging.getLogger(__name__)

def find_latest_checkpoint(dir: str) -> str:
    runs_dir = Path(dir)
    log.info(f"Searching for checkpoints in: {runs_dir}")

    if not runs_dir.exists():
        raise FileNotFoundError(f"The directory {runs_dir} does not exist.")

    # Get all run directories
    run_dirs = sorted(glob.glob(str(runs_dir / "*")), key=os.path.getctime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")

    # Get current timestamp
    current_time = datetime.now()

    # Search for checkpoints in all run directories, excluding directories created in the last minute
    all_checkpoints = []
    for run_dir in run_dirs:
        dir_creation_time = datetime.fromtimestamp(os.path.getctime(run_dir))
        if (current_time - dir_creation_time).total_seconds() < 60:  # Skip directories created in the last minute
            continue
        
        checkpoint_dir = Path(run_dir) / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = glob.glob(str(checkpoint_dir / "*.ckpt"))
            all_checkpoints.extend(checkpoints)

    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in any run directory under {runs_dir}")

    # Get the latest checkpoint
    latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
    log.info(f"Latest checkpoint file: {latest_checkpoint}")

    return latest_checkpoint
