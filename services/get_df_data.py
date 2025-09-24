import pandas as pd
import numpy as np
import datetime
import random
from pathlib import Path
from dateutil.relativedelta import relativedelta

DATA_DIR = Path("/app/data")

# ---- Lazy load datasets ----
_ckd_data_df = None
_ckd_crf_demo = None
_ordered_feats = None
_features_all_csn = None
_acr_df_pats = None
_look_up_p = None

def get_ckd_data():
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv",
                                   skipinitialspace=True, delimiter=",")
    return _ckd_data_df

def get_ckd_crf_demo():
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv", delimiter=",")
    return _ckd_crf_demo

def get_ordered_feats():
    global _ordered_feats
    if _ordered_feats is None:
        _ordered_feats = pd.read_csv(DATA_DIR / "ordered_feats.csv", delimiter=",")
    return _ordered_feats

def get_features_all_csn():
    global _features_all_csn
    if _features_all_csn is None:
        _features_all_csn = pd.read_csv(DATA_DIR / "features_all_csn_id.csv",
                                        skipinitialspace=True)
    return _features_all_csn

def get_acr_df():
    global _acr_df_pats
    if _acr_df_pats is None:
        _acr_df_pats = pd.read_csv(DATA_DIR / "cal_risk.csv", delimiter=",")
    return _acr_df_pats

def get_lookup_p():
    global _look_up_p
    if _look_up_p is None:
        _look_up_p = pd.read_csv(DATA_DIR / "look_up_p.csv")
    return _look_up_p


# ---- Public API functions (kept for compatibility) ----

def get_pat_kidney_risk(pat_id):
    df = get_acr_df()
    return df[df['pat_id'] == pat_id]

def get_pat_records(pat_id):
    df = get_ckd_data()
    return df[df['pat.id'] == pat_id]

def get_profile_date(pat_id):
    ages = get_pat_records(pat_id).age
    if len(ages) == 0:
        return ["1970-01-01", "1970-01-02"]
    cur_age = int(max(ages))
    in_date = '2015-06-15'
    dt = datetime.datetime.strptime(in_date, "%Y-%m-%d")
    n = random.randrange(360)
    last_visit_res = (dt + datetime.timedelta(days=n)).strftime("%Y-%m-%d")
    last_visit_date = datetime.datetime.strptime(last_visit_res,"%Y-%m-%d")
    rand_days = random.randrange(180)
    birth_day = (last_visit_date - relativedelta(years=cur_age)).strftime("%Y-%m-%d")
    return [birth_day, last_visit_res]

def get_pat_unique_concept(pat_id):
    df = get_pat_records(pat_id).sort_values(by=['age'])
    res_list = []
    for concept in df['concept.cd'].unique():
        ages = list(df[df['concept.cd'] == concept]['age'])
        vals = list(df[df['concept.cd'] == concept]['nval.num'])
        age_val = list(zip(ages, vals))
        res_list.append([concept, age_val])
    return res_list

def get_Umap_color(attr):
    df = get_ordered_feats()
    return list(df[attr])

def get_pat_demo(pat_id):
    df = get_ckd_crf_demo()
    return df[df['pat_id'] == pat_id]

def get_df_concept(att_name):
    df = get_ckd_data()
    return df[df['concept.cd'] == att_name]['nval.num']

def get_df_all_pat():
    df = get_ckd_data()
    return list(df['pat.id'].unique())

def get_df_all_concept():
    df = get_ckd_data()
    return list(df['concept.cd'].unique())

# --- Lab test related helpers ---

def get_pat_age_concept_list(pat_id):
    df = get_pat_records(pat_id).sort_values(by=['age'])
    concepts = df['concept.cd'].unique()
    ages = toIntegers(df['age'].unique())
    return list(range_list(ages[0], ages[-1]+1)), list(concepts)

def getLabTestViewData(pat_id):
    ages, pat_concepts = get_pat_age_concept_list(pat_id)
    concepts = get_df_all_concept()
    concept_age_val = get_pat_unique_concept(pat_id)
    res = []
    for concept_pair in concept_age_val:
        concept_ind = concepts.index(concept_pair[0])
        age_num_val_dict = {}
        for age_val_pair in concept_pair[1]:
            age = int(age_val_pair[0])
            val = age_val_pair[1]
            if age not in age_num_val_dict:
                age_num_val_dict[age] = [1, val]
            else:
                pre = age_num_val_dict[age]
                val_max = max(val, pre[1])
                age_num_val_dict[age] = [pre[0]+1, val_max]
        for key in age_num_val_dict:
            res.append([ages.index(key), concept_ind,
                        age_num_val_dict[key][0], age_num_val_dict[key][1]])
    return res


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

