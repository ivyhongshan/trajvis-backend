# scripts/offline_prepare.py
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
import umap.umap_ as umap
import simpleppt

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---------- 1) ?? UMAP ??? ----------
print("[prepare] load graphsage_output.npy ...")
output = np.load(DATA_DIR / "graphsage_output.npy")
df = pd.DataFrame(output)

print("[prepare] fit UMAP ...")
trans = umap.UMAP(
    n_neighbors=15, min_dist=1e-10, n_components=2,
    random_state=123, metric="euclidean", local_connectivity=1, verbose=1
).fit(df)

embedding = trans.embedding_
dump(trans, ARTIFACT_DIR / "umap_model.joblib")
np.save(ARTIFACT_DIR / "embedding.npy", embedding)
print("[prepare] saved umap_model.joblib, embedding.npy")

# ---------- 2) simpleppt????? run-time ?? get_ppt_trajectory? ----------
print("[prepare] build simpleppt ...")
ppt = simpleppt.ppt(embedding, Nodes=100, seed=1, progress=False, lam=200, sigma=0.3)
# ??????????? from/to ???????????????? ppt
# ???????????from/to ?????
from_list, to_list = [], []
for i in range(len(ppt.B)):
    for j in range(len(ppt.B[i])):
        if ppt.B[i][j] == 1:
            from_list.append(ppt.F.T[i].tolist())
            to_list.append(ppt.F.T[j].tolist())
edges_df = pd.DataFrame({"from": from_list, "to": to_list})
edges_df.to_parquet(ARTIFACT_DIR / "ppt_edges.parquet")
print("[prepare] saved ppt_edges.parquet")

# ---------- 3) ?? pat_traj.csv?????? ----------
print("[prepare] compute pat_traj.csv ...")
greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline  = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline= [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

features_all_csn = pd.read_csv(DATA_DIR / "features_all_csn_id.csv", skipinitialspace=True)

# ??????? pat_id ???????
def decide_traj(labels):
    s = set(labels)  # ?? set ??? in ????? labels ???
    orange = sum(1 for x in labels if x in orangeline)
    blue   = sum(1 for x in labels if x in blueline)
    green  = sum(1 for x in labels if x in greenline)
    if orange > blue and orange > green:
        return 'orange'
    elif blue > orange and blue > green:
        return 'blue'
    else:
        return 'green'

res = (
    features_all_csn
      .groupby("pat_id")["cluster_label"]
      .apply(list)
      .apply(decide_traj)
      .reset_index()
      .rename(columns={"cluster_label":"traj"})
)
res.to_csv(ARTIFACT_DIR / "pat_traj.csv", index=False)
print("[prepare] saved pat_traj.csv")

print("[prepare] ALL DONE. Artifacts at:", ARTIFACT_DIR)

