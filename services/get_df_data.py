# services/get_df_data.py
import pandas as pd
import datetime
import random
import os
from dateutil.relativedelta import relativedelta
from pathlib import Path
import numpy as np
#DATA_DIR = Path("/app/data")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
# ????
_ckd_data_df = None
_ckd_crf_demo = None
_features_all_csn = None
_acr_df_pats = None



def load_ckd_data_df():
    global _ckd_data_df
    if _ckd_data_df is None:
        print("Loading ckd_emr_data.csv ...")   # debug log
        print("DEBUG: entering load_ckd_data_df", flush=True)
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
    
def get_df_concept(att_name):
    df = load_ckd_data_df()
    return df[df['concept.cd'] == att_name]['nval.num']

def get_df_all_concept():
    df = load_ckd_data_df()
    return df['concept.cd'].unique().tolist()
    
def get_Umap_color(pat_id):
    df = get_pat_records(pat_id)
    if "EGFR" in df["concept.cd"].values:
        return df[df["concept.cd"] == "EGFR"]["nval.num"].tolist()
    else:
        return df["age"].tolist()
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
    
    
# --- UMAP color helpers ---
_umap_colors = None

def get_Umap_color():
    """
    Return two arrays: one for 'age' and one for 'EGFR'.
    Each array?? = ?????????.
    """
    global _umap_colors
    if _umap_colors is not None:
        return _umap_colors

    df = load_ckd_data_df()

    # ?????? age / EGFR ???????????
    age_map = df.groupby("pat.id")["age"].mean().to_dict()
    egfr_map = (
        df[df["concept.cd"] == "EGFR"]
        .groupby("pat.id")["nval.num"]
        .mean()
        .to_dict()
    )

    ages = [age_map.get(pid, None) for pid in df["pat.id"].unique()]
    egfrs = [egfr_map.get(pid, None) for pid in df["pat.id"].unique()]

    _umap_colors = (ages, egfrs)
    return _umap_colors


# ---------------- LabTest view ----------------
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
                val_max = val if val > pre[1] else pre[1]
                age_num_val_dict[age] = [pre[0] + 1, val_max]
        for age in age_num_val_dict:
            age_ind = ages.index(age)
            res.append([age_ind, concept_ind, age_num_val_dict[age][0], age_num_val_dict[age][1]])
    return res


# ---------------- Hierarchical clustering ----------------
def getHierarchicalClusterVec(pat_id):
    ages, concepts = get_pat_age_concept_list(pat_id)
    concept_age_val = get_pat_unique_concept(pat_id)
    concept_vec_dict = {}
    for concept_pair in concept_age_val:
        age_num_val_dict = {}
        for age_val_pair in concept_pair[1]:
            age = int(age_val_pair[0])
            if age not in age_num_val_dict:
                age_num_val_dict[age] = [1]
            else:
                age_num_val_dict[age][0] += 1
        res = []
        for age_uni in ages:
            res.append(1 if age_uni in age_num_val_dict else 0)
        concept_vec_dict[concept_pair[0]] = res
    return concept_vec_dict


def getHierarchicalClusterInput(pat_id):
    vect_dict = getHierarchicalClusterVec(pat_id)
    return list(vect_dict.values())


def getOrderofConcepts(pat_id):
    matrix = getHierarchicalClusterInput(pat_id)
    ages, concepts = get_pat_age_concept_list(pat_id)
    concepts_all = get_df_all_concept()
    if len(matrix) <= 1:
        return concepts
    model = AgglomerativeClustering(linkage="ward", distance_threshold=2, n_clusters=None)
    labels = model.fit_predict(matrix)
    key_tuples = [(c, l) for c, l in zip(concepts, labels)]
    newlist = sorted(key_tuples, key=lambda x: x[1], reverse=True)
    res = [i[0] for i in newlist]
    for c in concepts_all:
        if c not in res:
            res.append(c)
    return res


