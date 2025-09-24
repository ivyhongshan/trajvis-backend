import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


def load_features_all_csn():
    return pd.read_csv(DATA_DIR / "features_all_csn_id.csv", skipinitialspace=True)

greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline  = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline= [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

features_all_csn = load_features_all_csn()

rows = []
for pat_id in features_all_csn.pat_id.unique():
    individual_df = features_all_csn[features_all_csn['pat_id'] == pat_id]
    orange_num = sum(individual_df.cluster_label.isin(orangeline))
    blue_num   = sum(individual_df.cluster_label.isin(blueline))
    green_num  = sum(individual_df.cluster_label.isin(greenline))

    if orange_num > blue_num and orange_num > green_num:
        traj = 'orange'
    elif blue_num > orange_num and blue_num > green_num:
        traj = 'blue'
    else:
        traj = 'green'

    rows.append({'pat_id': pat_id, 'traj': traj})

res_df = pd.DataFrame(rows)
res_df.to_csv(DATA_DIR / "pat_traj.csv", index=False)

