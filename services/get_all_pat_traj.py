import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")

_pat_traj = None
def load_pat_traj():
    global _pat_traj
    if _pat_traj is None:
        _pat_traj = pd.read_csv(DATA_DIR / "pat_traj.csv")
    return _pat_traj

