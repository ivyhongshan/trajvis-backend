import pandas as pd
from pathlib import Path
import os
import time, logging

# DATA_DIR = Path("/app/data")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
_look_up_p = None


def load_look_up_p():
    global _look_up_p
    if _look_up_p is None:
        t0 = time.time()
        logging.info("Loading look_up_p.csv ...")
        _look_up_p = pd.read_csv(DATA_DIR / "look_up_p.csv")
        logging.info(f"look_up_p.csv loaded in {time.time()-t0:.2f}s")
    return _look_up_p