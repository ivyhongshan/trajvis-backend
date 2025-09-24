import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ?????Cloud Run ???? /app/data?
DATA_DIR = Path("/app/data")
ARTIFACT_DIR = DATA_DIR / "artifacts"

# ---- ?????? ----
_ckd_data_df = None
_ckd_crf_demo = None
_pat_traj = None
_trajectory_points = None
_neigh_graphsage = None


def get_ckd_data():
    """?? CKD EMR ??"""
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv", delimiter=",", skipinitialspace=True)
    return _ckd_data_df


def get_ckd_crf_demo():
    """?? CKD CRF ?????"""
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv", delimiter=",")
    return _ckd_crf_demo


def get_pat_traj():
    """?????? (pat_traj.csv)"""
    global _pat_traj
    if _pat_traj is None:
        _pat_traj = pd.read_csv(DATA_DIR / "pat_traj.csv", delimiter=",")
    return _pat_traj


def get_trajectory_points():
    """?? prepare_analysis.py ??????????"""
    global _trajectory_points
    if _trajectory_points is None:
        _trajectory_points = pd.read_csv(ARTIFACT_DIR / "trajectory_points.csv")
    return _trajectory_points


def get_neigh_graphsage():
    """?? prepare_analysis.py ???? KNN ??"""
    global _neigh_graphsage
    if _neigh_graphsage is None:
        _neigh_graphsage = joblib.load(ARTIFACT_DIR / "neigh_graphsage.joblib")
    return _neigh_graphsage

