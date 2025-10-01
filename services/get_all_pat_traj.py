import pandas as pd
from pathlib import Path
import os
import time, logging

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))


_pat_traj = None

def load_pat_traj():
    global _pat_traj
    if _pat_traj is None:
        t0 = time.time()
        logging.info("Loading pat_traj.csv ...")
        _pat_traj = pd.read_csv(DATA_DIR / "pat_traj.csv")
        logging.info(f"pat_traj.csv loaded in {time.time()-t0:.2f}s")
    return _pat_traj

