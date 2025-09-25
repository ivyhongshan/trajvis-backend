# services/get_analysis.py
import os
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
from sklearn.neighbors import KNeighborsClassifier

# ----------------
# 路径与全局缓存（lazy load）
# ----------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
ARTIFACT_DIR = DATA_DIR / "artifacts"

_ckd_data_df = None
_ckd_crf_demo = None
_pat_traj = None
_features_all_csn = None
_embeddings_all_id = None
_ordered_feats = None
_outputval_try = None
_neigh_graphsage = None

# ------------- Lazy loaders -------------
def get_ckd_data():
    global _ckd_data_df
    if _ckd_data_df is None:
        _ckd_data_df = pd.read_csv(DATA_DIR / "ckd_emr_data.csv", delimiter=",", skipinitialspace=True)
    return _ckd_data_df

def get_ckd_crf_demo():
    global _ckd_crf_demo
    if _ckd_crf_demo is None:
        _ckd_crf_demo = pd.read_csv(DATA_DIR / "ckd_crf_demo.csv", delimiter=",")
    return _ckd_crf_demo

def get_pat_traj():
    global _pat_traj
    if _pat_traj is None:
        _pat_traj = pd.read_csv(DATA_DIR / "pat_traj.csv", delimiter=",")
    return _pat_traj

def get_features_all_csn():
    global _features_all_csn
    if _features_all_csn is None:
        _features_all_csn = pd.read_csv(DATA_DIR / "features_all_csn_id.csv", delimiter=",")
    return _features_all_csn

def get_embeddings_all_id():
    global _embeddings_all_id
    if _embeddings_all_id is None:
        _embeddings_all_id = pd.read_csv(DATA_DIR / "embeddings_all_id_cluster.csv", delimiter=",")
    return _embeddings_all_id

def get_ordered_feats():
    global _ordered_feats
    if _ordered_feats is None:
        _ordered_feats = pd.read_csv(DATA_DIR / "ordered_feats.csv", delimiter=",")
    return _ordered_feats

def get_outputval_try():
    global _outputval_try
    if _outputval_try is None:
        arr = np.load(DATA_DIR / "graphsage_output.npy")
        _outputval_try = pd.DataFrame(arr)
    return _outputval_try

def get_neigh_graphsage():
    """
    优先从 artifacts 目录加载已经训练好的 KNN（neigh_graphsage.joblib）。
    若不存在，则按原始逻辑用 graphsage_output.npy + ordered_feats 重新 fit 一份 KNN。
    """
    global _neigh_graphsage
    if _neigh_graphsage is None:
        model_path = ARTIFACT_DIR / "neigh_graphsage.joblib"
        if model_path.exists():
            _neigh_graphsage = joblib.load(model_path)
        else:
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(get_outputval_try(), np.ravel(get_ordered_feats()["cluster_label"].values))
            _neigh_graphsage = neigh
    return _neigh_graphsage

# ----------------
# 原始曲线（保持原始逻辑）
# ----------------
def lowess(x, y, f=2. / 3., iter=3):
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

# 原始给定的三条 age/egfr 点（与你提供的版本一致）
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

green_age = [20.4301369863014, 31.3445205479452, 37.4541095890411, 42.2205479452055, 44.586301369863,
             49.71301369863015, 49.8883561643836, 48.7890410958904, 52.8123287671233, 54.7164383561644,
             59.7568493150685, 64.20958904109591, 61.8061643835616, 57.136301369863,
             67.9061643835616, 64.8356164383562, 69.1157534246575, 72.5178082191781,
             71.5691780821918, 71.8952054794521, 80.1205479452055, 77.2061643835616, 73.55,
             61.7486301369863]
green_egfr = [112.059529757143, 100.0341640733865, 97.1013982459755, 102.761026200269, 90.8680097939669,
              95.3956288032316, 102.774241653552, 102.688935636613, 98.7939364917503, 90.41361782967005,
              79.66282836588255, 69.0343322902359, 80.7122940246689, 84.29714296400769,
              53.1459257066338, 75.7299192666146, 67.2039465683612, 73.1266752511189,
              70.7468372579639, 57.609710695723706, 53.1720428523801, 45.1267891217517, 52.42981241600655,
              70.4927317098041]

def getFittingPoints(age, egfr, f=0.5):
    x_y = np.array(sorted(zip(age, egfr)))
    x, y = x_y[:, 0], x_y[:, 1]
    yest = lowess(x, y, f=f, iter=6)
    return [[round(a, 2), round(b, 2)] for a, b in zip(x, yest)]

def getTrajectoryPoints():
    return {
        'blue':  getFittingPoints(blue_age,   blue_egfr),
        'green': getFittingPoints(green_age,  green_egfr),
        'orange':getFittingPoints(orange_age, orange_egfr)
    }

# ----------------
# 统计分布（按 lazy 数据）
# ----------------
def get_pat_sex_distribution():
    res = []
    pat_traj = get_pat_traj()
    pat_demo = get_ckd_crf_demo()
    for traj in ['orange', 'blue', 'green']:
        pats = pat_traj[pat_traj['traj'] == traj].pat_id
        demo = pat_demo[pat_demo['pat_id'].isin(pats)]
        vc = demo.sex_cd.value_counts()
        F_num, M_num = vc.get('F', 0), vc.get('M', 0)
        tot = F_num + M_num
        res.append([traj, round(F_num / tot, 2) if tot else 0, round(M_num / tot, 2) if tot else 0])
    return res

def get_pat_race_distribution():
    res = []
    pat_traj = get_pat_traj()
    pat_demo = get_ckd_crf_demo()
    for traj in ['orange', 'blue', 'green']:
        pats = pat_traj[pat_traj['traj'] == traj].pat_id
        demo = pat_demo[pat_demo['pat_id'].isin(pats)]
        vc = demo.race_cd.value_counts()
        B_num, W_num = vc.get('B', 0), vc.get('W', 0)
        tot = B_num + W_num
        res.append([traj, round(B_num / tot, 2) if tot else 0, round(W_num / tot, 2) if tot else 0])
    return res

def get_concept_distribution(concept):
    x_value, y_value = [], []
    pat_traj = get_pat_traj()
    ckd_data_df = get_ckd_data()
    for traj in ['orange', 'blue', 'green']:
        pats = pat_traj[pat_traj['traj'] == traj].pat_id
        records = ckd_data_df[ckd_data_df['pat.id'].isin(pats)]
        target = records[records['concept.cd'] == concept]
        x_s, res = [], []
        i = 30
        while i < 90:
            x_s.append(i)
            value = target[(target['age'] > i) & (target['age'] < i + 5)]['nval.num'].mean()
            value = round(value, 2) if not pd.isna(value) else 0
            res.append(value)
            i += 5
        y_value.append([traj, res])
        x_value = x_s
    return x_value, y_value

# ----------------
# 曲线带宽辅助（与原始一致）
# ----------------
# 三类训练线索
greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline  = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline= [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

train_set = set(blueline) | set(orangeline) | set(greenline)

def pred_val(x, tck):
    return interpolate.splev(x, tck)

def getTwoPosition(input_xs, widths, tck):
    # 兼容“列表或标量”两种 widths 传参
    if not isinstance(widths, (list, tuple, np.ndarray)):
        widths = [widths] * len(input_xs)
    res = []
    for idx, input_x in enumerate(input_xs):
        width = widths[idx]
        x0, x1 = input_x - 0.1, input_x + 0.1
        y0, y1 = pred_val(x0, tck), pred_val(x1, tck)
        input_y = pred_val(input_x, tck)
        # 原始里 slope/切线计算最后没有用到，直接用垂直方向：
        theta = math.pi / 2
        output_x1 = input_x + math.cos(theta) * width / 2
        output_y1 = input_y + math.sin(theta) * width / 2
        output_x2 = input_x - math.cos(theta) * width / 2
        output_y2 = input_y - math.sin(theta) * width / 2
        res.append([(output_x1 + output_x2) / 2, output_y1, output_y2])
    return res

# ----------------
# 假想三条轨迹的“带宽点”生成（保持原始接口）
# ----------------
def fakeBlueData(inputx_s, widths):
    age, egfr = blue_age, blue_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)), decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, widths, tck)

def fakegreenData(inputx_s, widths):
    age, egfr = green_age, green_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)), decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, widths, tck)

def fakeOrangeData(inputx_s, widths):
    age, egfr = orange_age, orange_egfr
    data = np.round(np.array(getFittingPoints(age, egfr)), decimals=2)
    tck = interpolate.splrep(data[:, 0], data[:, 1])
    return getTwoPosition(inputx_s, widths, tck)

# ----------------
# 关键：按原始逻辑迁移的两个函数
# ----------------
def get_one_pat_cluster_label_by_age(id):
    """
    完整迁移原始逻辑（使用 lazy 数据源）:
    - 用 features_all_csn 定位该患者的 csn/age
    - 用 embeddings_all_id_cluster.csv 衔接 embedding 行
    - 若某 csn 在 ordered_feats 中出现，则直接取对应 cluster_label
      否则用 neigh_graphsage 预测
    - 计算三条轨迹在每个 age 的“带宽”（green/blue/orange），并向 80 岁外推
    返回: age_last, ages, green_width, blue_width, orange_width
    """
    features_all = get_features_all_csn()
    embeddings_all = get_embeddings_all_id()
    ordered_feats = get_ordered_feats()
    neigh = get_neigh_graphsage()

    features_id = features_all[features_all['pat_id'] == id]
    id_csn = list(features_id.csn)
    embedding_id = embeddings_all[embeddings_all['csn'].isin(id_csn)]
    ages = []
    for csn in embedding_id.csn:
        age = list(features_id[features_id.csn == csn].age)[0]
        ages.append(age)
    embedding_id = embedding_id.copy()
    embedding_id.insert(1, 'age', ages)
    embedding_id = embedding_id.sort_values('age')

    ppt_cluster_label = []
    for csn_num in embedding_id.csn:
        if csn_num in list(ordered_feats.csn):
            ppt_cluster_label.append(list(ordered_feats[ordered_feats.csn == csn_num].cluster_label)[0])
        else:
            temp = neigh.predict(embedding_id[embedding_id.csn == csn_num].iloc[:, 6:])[0]
            ppt_cluster_label.append(temp)

    green_label, blue_label, orange_label = [], [], []
    for label in ppt_cluster_label:
        blue_label.append(1 if label in blueline else 0)
        green_label.append(1 if label in greenline else 0)
        orange_label.append(1 if label in orangeline else 0)

    d = {
        'age': list(embedding_id['age']),
        'csn': list(embedding_id['csn']),
        'cluster_label': ppt_cluster_label,
        'green_label': green_label,
        'blue_label': blue_label,
        'orange_label': orange_label
    }
    pat_cluster = pd.DataFrame(d)

    ages_series = [int(list(pat_cluster.age)[0])]
    green_width = [33]
    blue_width  = [33]
    orange_width= [33]

    for _, row in pat_cluster.iterrows():
        label = row['cluster_label']
        if label in train_set:
            ages_series.append(round(row['age'], 2))
            num = row['green_label'] + row['blue_label'] + row['orange_label']
            res = 3 - num
            # green
            if row['green_label'] > 0:
                green_width.append(round(green_width[-1] + 1 / num, 2))
            else:
                green_width.append(0 if green_width[-1] < 0 else round(green_width[-1] - 1 / res, 2))
            # blue
            if row['blue_label'] > 0:
                blue_width.append(round(blue_width[-1] + 1 / num, 2))
            else:
                blue_width.append(0 if blue_width[-1] < 0 else round(blue_width[-1] - 1 / res, 2))
            # orange
            if row['orange_label'] > 0:
                orange_width.append(round(orange_width[-1] + 1 / num, 2))
            else:
                orange_width.append(0 if orange_width[-1] < 0 else round(orange_width[-1] - 1 / res, 2))

    j = ages_series[-1]
    age_last = ages_series[-1]
    plus = 0
    minors = 0
    orange_last = orange_width[-1]
    blue_last = blue_width[-1]
    green_last = green_width[-1]
    while j < 80:
        j = j + 1
        plus += 0.5
        minors += 1 / 4
        ages_series.append(j)
        if orange_last + plus > 100 or green_last + plus > 100 or blue_last + plus > 100:
            orange_width.append(orange_last)
            blue_width.append(blue_last)
            green_width.append(green_last)
            continue
        if orange_last >= blue_last and orange_last >= green_last:
            green_width.append(0 if green_width[-1] < 0 else green_last - minors)
            blue_width.append(0 if blue_width[-1] < 0 else blue_last - minors)
            orange_width.append(orange_last + plus)
        elif blue_last >= orange_last and blue_last >= green_last:
            green_width.append(0 if green_width[-1] < 0 else green_last - minors)
            blue_width.append(blue_last + plus)
            orange_width.append(0 if orange_width[-1] < 0 else orange_last - minors)
        elif green_last >= orange_last and green_last >= blue_last:
            green_width.append(green_last + plus)
            blue_width.append(0 if blue_width[-1] < 0 else blue_last - minors)
            orange_width.append(0 if orange_width[-1] < 0 else orange_last - minors)

    return age_last, ages_series, green_width, blue_width, orange_width


def get_three_uncertainty(id):
    """
    完整迁移原始逻辑（使用 lazy 数据源），与 get_one_pat_cluster_label_by_age 类似，
    但采用了 min/max 宽度上下界和不同的增减规则。
    返回: ages, green_width, blue_width, orange_width
    """
    features_all = get_features_all_csn()
    embeddings_all = get_embeddings_all_id()
    ordered_feats = get_ordered_feats()
    neigh = get_neigh_graphsage()

    features_id = features_all[features_all['pat_id'] == id]
    id_csn = list(features_id.csn)
    embedding_id = embeddings_all[embeddings_all['csn'].isin(id_csn)]
    ages = []
    for csn in embedding_id.csn:
        age = list(features_id[features_id.csn == csn].age)[0]
        ages.append(age)
    embedding_id = embedding_id.copy()
    embedding_id.insert(1, 'age', ages)
    embedding_id = embedding_id.sort_values('age')

    ppt_cluster_label = []
    for csn_num in embedding_id.csn:
        if csn_num in list(ordered_feats.csn):
            ppt_cluster_label.append(list(ordered_feats[ordered_feats.csn == csn_num].cluster_label)[0])
        else:
            temp = neigh.predict(embedding_id[embedding_id.csn == csn_num].iloc[:, 6:])[0]
            ppt_cluster_label.append(temp)

    green_label, blue_label, orange_label = [], [], []
    for label in ppt_cluster_label:
        blue_label.append(1 if label in blueline else 0)
        green_label.append(1 if label in greenline else 0)
        orange_label.append(1 if label in orangeline else 0)

    d = {
        'age': list(embedding_id['age']),
        'csn': list(embedding_id['csn']),
        'cluster_label': ppt_cluster_label,
        'green_label': green_label,
        'blue_label': blue_label,
        'orange_label': orange_label
    }
    pat_cluster = pd.DataFrame(d)

    ages_series = [int(list(pat_cluster.age)[0])]
    green_width = [30]
    blue_width  = [30]
    orange_width= [30]
    min_width = 10
    max_width = 50

    for _, row in pat_cluster.iterrows():
        label = row['cluster_label']
        if label in train_set:
            ages_series.append(round(row['age'], 2))
            num = row['green_label'] + row['blue_label'] + row['orange_label']
            # 与原始一致：出现该颜色则减小，不出现则增加
            if row['green_label'] > 0:
                green_width.append(min_width if green_width[-1] < min_width else round(green_width[-1] - 1 / num, 2))
            else:
                green_width.append(round(green_width[-1] + 1 / num, 2))
            if row['blue_label'] > 0:
                blue_width.append(min_width if blue_width[-1] < min_width else round(blue_width[-1] - 1 / num, 2))
            else:
                blue_width.append(round(blue_width[-1] + 1 / num, 2))
            if row['orange_label'] > 0:
                orange_width.append(min_width if orange_width[-1] < min_width else round(orange_width[-1] - 1 / num, 2))
            else:
                orange_width.append(round(orange_width[-1] + 1 / num, 2))

    j = ages_series[-1]
    plus = 0
    minors = 0
    orange_last = orange_width[-1]
    blue_last = blue_width[-1]
    green_last = green_width[-1]
    while j < 80:
        j = j + 1
        plus += 0.5
        minors += 1 / 4
        ages_series.append(j)
        # 边界判定
        if orange_last - plus < min_width or green_last - plus < min_width or blue_last - plus < min_width:
            green_width.append(green_last)
            orange_width.append(orange_last)
            blue_width.append(blue_last)
            continue
        # 最小者削减，其他增加（与原始保持一致的相对关系）
        if orange_last < blue_last and orange_last < green_last:
            green_width.append(green_last + minors if green_last < max_width else max_width)
            blue_width.append(blue_last + minors if blue_last < max_width else max_width)
            orange_width.append(10 if orange_last < 10 else orange_last - plus)
        elif blue_last < orange_last and blue_last < green_last:
            green_width.append(green_last + minors if green_last < max_width else max_width)
            orange_width.append(orange_last + minors if orange_last < max_width else max_width)
            blue_width.append(10 if blue_last < 10 else blue_last - plus)
        elif green_last < blue_last and green_last < orange_last:
            orange_width.append(orange_last + minors if orange_last < max_width else max_width)
            blue_width.append(blue_last + minors if blue_last < max_width else max_width)
            green_width.append(10 if green_last < 10 else green_last - plus)

        # 更新“last”
        orange_last = orange_width[-1]
        blue_last = blue_width[-1]
        green_last = green_width[-1]

    # 裁剪到 [min_width, max_width]
    green_width  = list(np.clip(green_width,  min_width, max_width))
    blue_width   = list(np.clip(blue_width,   min_width, max_width))
    orange_width = list(np.clip(orange_width, min_width, max_width))
    return ages_series, green_width, blue_width, orange_width


def getThreePossibility(ages, green_width, blue_width, orange_width):
    """
    与原始一致：按 age 聚合（平均），对缺失年龄插值，再对三列做行归一化，并 round(2)
    """
    merge_df = pd.DataFrame({
        'age': ages,
        'green_width': green_width,
        'blue_width': blue_width,
        'orange_width': orange_width
    })
    merge_df.age = merge_df.age.astype(int)

    agg = {'green_width': 'mean', 'blue_width': 'mean', 'orange_width': 'mean'}
    df_new = merge_df.groupby('age', as_index=False).aggregate(agg)

    # 补充缺失年龄 & 插值（按原始写法）
    # （这里保持最小起点 30，到 df_new.age.max()）
    for age in range(30, int(df_new.age.max())):
        if age < int(df_new.age.min()):
            df_new = pd.concat([df_new, pd.DataFrame([{'age': age, 'green_width': 0.33, 'blue_width': 0.33, 'orange_width': 0.33}])], ignore_index=True)
        elif age not in list(df_new.age):
            df_new = pd.concat([df_new, pd.DataFrame([{'age': age, 'green_width': np.nan, 'blue_width': np.nan, 'orange_width': np.nan}])], ignore_index=True)

    df_new = df_new.sort_values(by='age', ascending=True)
    df_new = df_new.interpolate()

    # 行归一化
    df_new.iloc[:, 1:] = df_new.iloc[:, 1:].div(df_new.iloc[:, 1:].sum(axis=1), axis=0)
    df_new = df_new.round(2)
    return list(df_new.age), list(df_new.green_width), list(df_new.blue_width), list(df_new.orange_width)
