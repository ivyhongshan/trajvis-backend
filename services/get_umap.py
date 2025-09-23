# services/get_umap.py
import logging
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import pandas as pd
from joblib import load, dump
import umap.umap_ as umap

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

greenline = [26,22,83,88,46,51,24,65,4,16,89,30,32,13,78,18,1,64,97,69,33,60,28,3,20,74,62,91,66,94,75,44,61,54]
blueline  = [26,22,83,88,46,51,24,65,4,16,89,30,32,13,78,18,1,64,97,69,33,0,99,58,29,47,82,67,14]
orangeline= [26,22,83,88,46,51,24,65,4,16,49,85,72,34,25,10,73,5,59]

# ???????????????

def _ensure_artifacts():
    """? artifacts ??????????????????????????????"""
    need_fit = not (ARTIFACT_DIR / "umap_model.joblib").exists()
    need_embed = not (ARTIFACT_DIR / "embedding.npy").exists()
    if need_fit or need_embed:
        logging.info("Artifacts missing; computing once...")
        output = np.load(DATA_DIR / "graphsage_output.npy")
        df = pd.DataFrame(output)
        trans = umap.UMAP(
            n_neighbors=15, min_dist=1e-10, n_components=2,
            random_state=123, metric="euclidean", local_connectivity=1, verbose=1
        ).fit(df)
        dump(trans, ARTIFACT_DIR / "umap_model.joblib")
        np.save(ARTIFACT_DIR / "embedding.npy", trans.embedding_)
        logging.info("Artifacts created.")

@lru_cache(maxsize=1)
def _state():
    _ensure_artifacts()
    logging.info("Loading artifacts...")
    st = {}
    st["trans"] = load(ARTIFACT_DIR / "umap_model.joblib")
    st["embedding"] = np.load(ARTIFACT_DIR / "embedding.npy")
    # ?? runtime ?????? CSV????????????
    st["embeddings_all_id_df"] = pd.read_csv(DATA_DIR / "embeddings_all_id_cluster.csv")
    st["features_all_csn_df"]  = pd.read_csv(DATA_DIR / "features_all_csn_id.csv")
    # ppt ????? get_ppt_trajectory??
    ppt_edges = ARTIFACT_DIR / "ppt_edges.parquet"
    st["ppt_edges_df"] = pd.read_parquet(ppt_edges) if ppt_edges.exists() else None
    # pat_traj
    pat_traj = ARTIFACT_DIR / "pat_traj.csv"
    st["pat_traj_df"] = pd.read_csv(pat_traj) if pat_traj.exists() else None
    return st

def warm():
    _ = _state(); return True

def get_orginal_embed():
    return _state()["embedding"].tolist()

def tranform_new_data(data):
    trans = _state()["trans"]
    return trans.transform(data)

def project_to_umap(id):
    st = _state()
    latent_df = st["embeddings_all_id_df"][st["embeddings_all_id_df"]["pat_id"] == id]
    clean_latent_df = latent_df.iloc[:, 5:]
    return tranform_new_data(clean_latent_df)

def get_pat_age_egfr(pat_id):
    st = _state()
    emb_df = st["embeddings_all_id_df"]
    feats_df = st["features_all_csn_df"]
    csn_list = list(emb_df[emb_df["pat_id"] == pat_id].csn)
    ages, egfrs = [], []
    for csn in csn_list:
        row = feats_df[feats_df["csn"] == csn].iloc[0]
        ages.append(row["age"])
        egfrs.append(row["EGFR_val"])
    return ages, egfrs

def get_ppt_trajectory():
    st = _state()
    df = st["ppt_edges_df"]
    if df is None:
        # ???? ppt_edges??????????
        return pd.DataFrame(columns=["from","to"])
    return df

def get_four_trajectory():
    # ???????????
    return [["green", green_traj_smooth], ["blue", blue_traj_smooth], ["orange", orange_traj_smooth]]

