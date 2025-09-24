import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = Path("data")   # ????? data/
ARTIFACT_DIR = DATA_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
features_all_csn_df = pd.read_csv(DATA_DIR / "features_all_csn_id.csv")
embeddings_all_id_df = pd.read_csv(DATA_DIR / "embeddings_all_id_cluster.csv")
ordered_feats = pd.read_csv(DATA_DIR / "ordered_feats.csv")
outputval_try = np.load(DATA_DIR / "graphsage_output.npy")
outputval_try = pd.DataFrame(outputval_try)

# ---- Train & save KNN ----
neigh_graphsage = KNeighborsClassifier(n_neighbors=5)
neigh_graphsage.fit(outputval_try, np.ravel(ordered_feats['cluster_label'].values))
joblib.dump(neigh_graphsage, ARTIFACT_DIR / "neigh_graphsage.joblib")

# ---- Example: save trajectory points ----
def lowess(x, y, f=2./3., iter=3):
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return yest

blue_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            59.7568493150685, 64.20958904109591, 61.8061643835616, 57.136301369863, 
            67.9061643835616, 68.77397260273969, 69.6445205479452, 71.1650684931507, 
            70.5671232876712]
blue_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            79.66282836588255, 69.0343322902359, 80.7122940246689, 84.29714296400769, 
            53.1459257066338, 49.71462777904635, 42.3471510341773, 45.224953443202, 
            32.78494936494355]

orange_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            49.1705479452055, 54.4945205479452, 56.8164383561644, 55.9390410958904, 63.6404109589041, 
            64.9424657534247, 67.9061643835616]
orange_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            95.8349884573413, 96.9980674157135, 89.0202724482197, 67.1871752285721, 56.5887043081018, 
            49.273873430824096, 53.1459257066338]

# del 14 15
green_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863, 
            49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644, 
            59.7568493150685, 64.20958904109591, 61.8061643835616, 57.136301369863,  
            67.9061643835616, 64.8356164383562, 69.1157534246575, 72.5178082191781, 
            71.5691780821918, 71.8952054794521, 80.1205479452055, 77.2061643835616, 73.55000000000001, 
            61.7486301369863]
green_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669, 
            95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005, 
            79.66282836588255, 69.0343322902359, 80.7122940246689, 84.29714296400769, 
            53.1459257066338, 75.7299192666146, 67.2039465683612, 73.1266752511189, 
            70.7468372579639, 57.609710695723706, 53.1720428523801, 45.1267891217517, 52.42981241600655, 
            70.4927317098041]
def getFittingPoints(age, egfr):
    x_y = sorted(zip(age, egfr))
    x, y = np.array(x_y)[:,0], np.array(x_y)[:,1]
    yest = lowess(x, y, f=0.5, iter=6)
    return [(round(float(xx),2), round(float(yy),2)) for xx,yy in zip(x,yest)]

trajectory = {
    "blue": getFittingPoints(blue_age, blue_egfr),
    "green": getFittingPoints(green_age, green_egfr),
    "orange": getFittingPoints(orange_age, orange_egfr)
}
pd.DataFrame({k: [v] for k,v in trajectory.items()}).to_csv(ARTIFACT_DIR / "trajectory_points.csv", index=False)
print("Saved trajectory_points.csv and neigh_graphsage.joblib")

