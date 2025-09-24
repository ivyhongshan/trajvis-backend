# services/get_df_data.py
import pandas as pd
import datetime
import random
from dateutil.relativedelta import relativedelta
from pathlib import Path

DATA_DIR = Path("/app/data")

# ????
_ckd_data_df = None
_ckd_crf_demo = None
_features_all_csn = None
_acr_df_pats = None

def load_ckd_data_df():
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv", skipinitialspace=True)
    return _ckd_data_df

def load_ckd_crf_demo():
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv")
    return _ckd_crf_demo

def load_features_all_csn():
    global _features_all_csn
    if _features_all_csn is None:
        _features_all_csn = pd.read_csv(DATA_DIR / "features_all_csn_id.csv", skipinitialspace=True)
    return _features_all_csn

def load_acr_df_pats():
    global _acr_df_pats
    if _acr_df_pats is None:
        _acr_df_pats = pd.read_csv(DATA_DIR / "cal_risk.csv")
    return _acr_df_pats

# ---------------- ???? ----------------
def get_pat_records(pat_id):
    df = load_ckd_data_df()
    return df[df["pat.id"] == pat_id]

def get_df_all_pat():
    df = load_ckd_data_df()
    return df["pat.id"].unique().tolist()

def get_pat_demo(pat_id):
    df = load_ckd_crf_demo()
    return df[df["pat_id"] == pat_id]

def get_pat_unique_concept(pat_id):
    df = get_pat_records(pat_id).sort_values(by="age")
    concepts = df["concept.cd"].unique()
    res_list = []
    for c in concepts:
        ages = df[df["concept.cd"] == c]["age"].tolist()
        vals = df[df["concept.cd"] == c]["nval.num"].tolist()
        age_val = list(map(lambda x, y: [x, y], ages, vals))
        res_list.append([c, age_val])
    return res_list

def get_pat_kidney_risk(pat_id):
    df = load_acr_df_pats()
    return df[df["pat_id"] == pat_id]

def get_profile_date(pat_id):
    ages = get_pat_records(pat_id)["age"]
    if len(ages) == 0:
        return None
    cur_age = int(max(ages))
    in_date = "2015-6-15"
    dt = datetime.datetime.strptime(in_date, "%Y-%m-%d")
    n = random.randrange(360)
    last_visit_res = (dt + datetime.timedelta(days=n)).strftime("%Y-%m-%d")
    last_visit_date = datetime.datetime.strptime(last_visit_res, "%Y-%m-%d")
    rand_days = random.randrange(180)

    birth_day = (last_visit_date - relativedelta(years=cur_age)).strftime("%Y-%m-%d")
    birth_day = datetime.datetime.strptime(birth_day, "%Y-%m-%d")
    birth_day = (birth_day - datetime.timedelta(days=rand_days)).strftime("%Y-%m-%d")

    return [birth_day, last_visit_res]


# --- Normal range dict (unchanged) ---
normal_range_dict = {
    'EGFR':[60, 200],
    'TBIL': [0.1, 1.2],
    'BP_DIASTOLIC': [60, 80],
    'BP_SYSTOLIC': [90, 120],
    'WT': [90, 220],
    'HT': [57, 78],
    'CHOLESTEROL': [50, 200],
    'CREATINE_KINASE': [22, 198],
    'HEMOGLOBIN': [11.6, 17.2],
    'INR': [0.8, 1.1],
    'ALT_SGPT': [7, 56],
    'AST_SGOT': [8, 45],
    'ALK': [44, 147],
    'HDL': [40, 100],
    'LDL': [40, 100],
    'TRIGLYCERIDES': [20, 150],
    'HBA1C': [4, 6.5],
    'TROPONIN': [0, 0.04]
}

# --- Utilities ---
def toIntegers(data):
    return np.trunc(data).astype(int)

def range_list(a, b):
    return list(range(a, b+1))

