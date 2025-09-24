import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")

_look_up_p = None

def load_look_up_p():
    global _look_up_p
    if _look_up_p is None:
        _look_up_p = pd.read_csv(DATA_DIR / "look_up_p.csv")
    return _look_up_p

