import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ----------------
# ????
# ----------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
ARTIFACT_DIR = DATA_DIR / "artifacts"

# ----------------
# Lazy global cache
# ----------------
_ckd_data_df = None
_ckd_crf_demo = None
_pat_traj = None
_trajectory_points = None
_neigh_graphsage = None


def get_ckd_data():
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv", delimiter=",")
    return _ckd_data_df


def get_ckd_crf_demo():
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv", delimiter=",")
    return _ckd_crf_demo


def get_pat_traj():
    global _pat_traj
    if _pat_traj is None:
        _pat_traj = pd.read_csv(DATA_DIR / "pat_traj.csv", delimiter=",")
    return _pat_traj


def get_trajectory_points():
    """???????? trajectory_points.csv"""
    global _trajectory_points
    if _trajectory_points is None:
        _trajectory_points = pd.read_csv(ARTIFACT_DIR / "trajectory_points.csv")
    return _trajectory_points


def get_neigh_graphsage():
    """??????? KNN ??"""
    global _neigh_graphsage
    if _neigh_graphsage is None:
        _neigh_graphsage = joblib.load(ARTIFACT_DIR / "neigh_graphsage.joblib")
    return _neigh_graphsage


# ----------------
# ??????
# ----------------

def getTrajectoryPoints():
    """??????????"""
    df = get_trajectory_points()
    result = {}
    for traj in df["traj"].unique():
        sub = df[df["traj"] == traj][["age", "egfr"]]
        result[traj] = sub.values.tolist()
    return result


def get_pat_sex_distribution():
    """????????"""
    pat_traj = get_pat_traj()
    demo = get_ckd_crf_demo()
    res = []
    for traj in ["orange", "blue", "green"]:
        pats = pat_traj[pat_traj["traj"] == traj].pat_id
        pat_demo = demo[demo["pat_id"].isin(pats)]
        sex_value_counts = pat_demo.sex_cd.value_counts()
        F_num = sex_value_counts.get("F", 0)
        M_num = sex_value_counts.get("M", 0)
        total = F_num + M_num
        if total == 0:
            res.append([traj, 0, 0])
        else:
            res.append([traj, round(F_num / total, 2), round(M_num / total, 2)])
    return res


def get_pat_race_distribution():
    """????????"""
    pat_traj = get_pat_traj()
    demo = get_ckd_crf_demo()
    res = []
    for traj in ["orange", "blue", "green"]:
        pats = pat_traj[pat_traj["traj"] == traj].pat_id
        pat_demo = demo[demo["pat_id"].isin(pats)]
        race_value_counts = pat_demo.race_cd.value_counts()
        B_num = race_value_counts.get("B", 0)
        W_num = race_value_counts.get("W", 0)
        total = B_num + W_num
        if total == 0:
            res.append([traj, 0, 0])
        else:
            res.append([traj, round(B_num / total, 2), round(W_num / total, 2)])
    return res


def get_concept_distribution(concept):
    """??????????????"""
    pat_traj = get_pat_traj()
    ckd_data = get_ckd_data()
    x_value = []
    y_value = []
    for traj in ["orange", "blue", "green"]:
        pats = pat_traj[pat_traj["traj"] == traj].pat_id
        records = ckd_data[ckd_data["pat.id"].isin(pats)]
        target = records[records["concept.cd"] == concept]
        x_s = []
        res = []
        i = 30
        while i < 90:
            x_s.append(i)
            value = target[(target["age"] > i) & (target["age"] < i + 5)]["nval.num"].mean()
            value = round(value, 2) if not pd.isna(value) else 0
            res.append(value)
            i += 5
        y_value.append([traj, res])
        x_value = x_s
    return x_value, y_value

