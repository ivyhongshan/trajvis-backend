# services/get_umap.py
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
import umap

# Data ??????? & Cloud Run?
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
ARTIFACT_DIR = DATA_DIR / "artifacts"

# --- fixed trajectory lines ---
greenline = [26,22,83,88,46,51,24,65,4,16,89,30,32,13,78,18,1,64,97,69,33,60,28,3,20,74,62,91,66,94,75,44,61,54]
blueline  = [26,22,83,88,46,51,24,65,4,16,89,30,32,13,78,18,1,64,97,69,33,0,99,58,29,47,82,67,14]
orangeline= [26,22,83,88,46,51,24,65,4,16,49,85,72,34,25,10,73,5,59]

# --- pre-saved smoothed trajectories ---
# (???????? green_traj_smooth / blue_traj_smooth / orange_traj_smooth)
orange_traj_smooth = [[4.99, -1.67],[5.26, -1.52],[5.63, -1.31],[5.96, -1.12],[6.27, -0.93],
 [6.36, -0.36],[6.13, -0.02],[6.02, 0.37],[6.04, 0.99],[5.53, 1.41],[5.33, 1.93],
 [5.20, 2.37],[4.84, 2.36]]

green_traj_smooth = [[4.99, -1.67],[5.26, -1.52],[5.63, -1.31],[5.96, -1.12],[6.27, -0.93],
 [6.36, -0.36],[6.13, -0.02],[6.02, 0.37],[6.04, 0.99],[6.59, 1.36],[7.13, 1.68],
 [7.37, 1.02],[7.58, 0.43],[8.01, 0.56],[8.50, 0.71],[8.87, 1.29],[8.66, 2.05],
 [8.68, 2.62],[8.92, 2.97],[9.30, 3.21],[9.57, 3.53],[9.58, 4.02],[8.96, 4.89],
 [8.02, 4.77],[7.45, 4.78]]

blue_traj_smooth = [[4.99, -1.67],[5.26, -1.52],[5.63, -1.31],[5.96, -1.12],[6.27, -0.93],
 [6.36, -0.36],[6.13, -0.02],[6.02, 0.37],[6.04, 0.99],[6.59, 1.36],[7.13, 1.68],
 [7.37, 1.02],[7.58, 0.43],[8.01, 0.56],[8.50, 0.71],[8.87, 1.29],[9.54, 0.97],
 [10.08, 0.71],[10.57, 0.48],[11.00, 0.32],[11.41, 0.55],[11.59, 0.78],[11.64, 0.91]]

# --- Ensure artifacts exist ---
def _ensure_artifacts():
    """?? artifacts ??????????"""
    if not (ARTIFACT_DIR / "umap_model.joblib").exists():
        raise FileNotFoundError("Missing umap_model.joblib in artifacts dir")
    if not (ARTIFACT_DIR / "embedding.npy").exists():
        raise FileNotFoundError("Missing embedding.npy in artifacts dir")

@lru_cache(maxsize=1)
def _state():
    t0 = time.time()
    _ensure_artifacts()
    logging.info("Loading artifacts in get_umap...")

    st = {}
    st["trans"] = load(ARTIFACT_DIR / "umap_model.joblib")
    logging.info(f"umap_model.joblib loaded in {time.time()-t0:.2f}s")

    st["embedding"] = np.load(ARTIFACT_DIR / "embedding.npy")
    logging.info(f"embedding.npy loaded in {time.time()-t0:.2f}s")

    st["embeddings_all_id_df"] = pd.read_csv(DATA_DIR / "embeddings_all_id_cluster.csv")
    logging.info(f"embeddings_all_id_cluster.csv loaded in {time.time()-t0:.2f}s")

    st["features_all_csn_df"] = pd.read_csv(DATA_DIR / "features_all_csn_id.csv")
    logging.info(f"features_all_csn_id.csv loaded in {time.time()-t0:.2f}s")

    total = time.time()-t0
    logging.info(f"get_umap._state total load time {total:.2f}s")
    return st


def warm():
    _ = _state(); return True

def get_orginal_embed():
    return _state()["embedding"].tolist()

def tranform_new_data(data):
    return _state()["trans"].transform(data)

def project_to_umap(pat_id):
    st = _state()
    latent_df = st["embeddings_all_id_df"][st["embeddings_all_id_df"]["pat_id"] == pat_id]
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
    # ????? simpleppt ????? runtime ????
    return pd.DataFrame(columns=["from", "to"])

def get_four_trajectory():
    # ?? smooth ??
    return [
        ["green", green_traj_smooth],
        ["blue",  blue_traj_smooth],
        ["orange", orange_traj_smooth]
    ]

