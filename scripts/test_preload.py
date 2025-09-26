import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.get_df_data import (
    load_ckd_data_df,
    load_ckd_crf_demo,
    load_features_all_csn,
    load_acr_df_pats,
)
from services.get_analysis import getTrajectoryPoints, get_neigh_graphsage


def timed_call(name, func):
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        print(f"[OK] {name} loaded in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] {name} raised {e} after {elapsed:.2f} seconds")


def main():
    print("=== Preload Test ===")

    timed_call("ckd_emr_data.csv", load_ckd_data_df)
    timed_call("ckd_crf_demo.csv", load_ckd_crf_demo)
    timed_call("features_all_csn_id.csv", load_features_all_csn)
    timed_call("cal_risk.csv", load_acr_df_pats)

    timed_call("trajectory_points.csv", getTrajectoryPoints)
    timed_call("neigh_graphsage.joblib", get_neigh_graphsage)

    print("=== Preload Test Finished ===")


if __name__ == "__main__":
    main()
