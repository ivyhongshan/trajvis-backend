import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")

_ckd_data_df = None
_ckd_crf_demo = None
_ordered_feats = None
_features_all_csn = None
_acr_df_pats = None
_look_up_p = None

def load_ckd_data():
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv", skipinitialspace=True, delimiter=",")
    return _ckd_data_df

def load_ckd_crf_demo():
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv", delimiter=",")
    return _ckd_crf_demo

def load_ordered_feats():
    global _ordered_feats
    if _ordered_feats is None:
        _ordered_feats = pd.read_csv(DATA_DIR / "ordered_feats.csv", delimiter=",")
    return _ordered_feats

def load_features_all_csn():
    global _features_all_csn
    if _features_all_csn is None:
        _features_all_csn = pd.read_csv(DATA_DIR / "features_all_csn_id.csv", delimiter=",", skipinitialspace=True)
    return _features_all_csn

def load_acr_df_pats():
    global _acr_df_pats
    if _acr_df_pats is None:
        _acr_df_pats = pd.read_csv(DATA_DIR / "cal_risk.csv", delimiter=",")
    return _acr_df_pats

def load_look_up_p():
    global _look_up_p
    if _look_up_p is None:
        _look_up_p = pd.read_csv(DATA_DIR / "look_up_p.csv")
    return _look_up_p

