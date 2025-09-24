# scripts/prepare_umap.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
import umap.umap_ as umap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphsage-npy", required=True, help="path to graphsage_output.npy")
    ap.add_argument("--out-dir", required=True, help="output dir (local)")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=1e-10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.graphsage_npy)
    df = pd.DataFrame(arr)

    um = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        random_state=123,
        metric="euclidean",
        local_connectivity=1,
        verbose=True,
    ).fit(df)

    np.save(out_dir / "embedding.npy", um.embedding_)
    dump(um, out_dir / "umap_model.joblib")

    print("Saved:", out_dir / "embedding.npy")
    print("Saved:", out_dir / "umap_model.joblib")

if __name__ == "__main__":
    main()

