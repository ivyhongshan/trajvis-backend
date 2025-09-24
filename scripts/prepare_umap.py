# scripts/prepare_umap.py
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
import umap

DATA_DIR = Path("data")
ARTIFACT_DIR = DATA_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # ?????graphsage ??
    graphsage_out = DATA_DIR / "graphsage_output.npy"
    output = np.load(graphsage_out)
    df = pd.DataFrame(output)

    # ?? UMAP
    trans = umap.UMAP(
        n_neighbors=15, min_dist=1e-10, n_components=2,
        random_state=123, metric="euclidean", local_connectivity=1, verbose=True
    ).fit(df)

    # ????
    dump(trans, ARTIFACT_DIR / "umap_model.joblib")
    np.save(ARTIFACT_DIR / "embedding.npy", trans.embedding_)

    print(f"? Saved: {ARTIFACT_DIR/'umap_model.joblib'}, {ARTIFACT_DIR/'embedding.npy'}")

if __name__ == "__main__":
    main()

