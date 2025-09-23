# services/get_umap.py
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import umap.umap_ as umap
import simpleppt

# ------- ???? verbose ????????? -------
UMAP_VERBOSE = 1  # ???? 3????????????? 3

# ??????????????
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ???????????????
greenline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 60, 28, 3, 20, 74, 62, 91, 66, 94, 75, 44, 61, 54]
blueline  = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 89, 30, 32, 13, 78, 18, 1, 64, 97, 69, 33, 0, 99, 58, 29, 47, 82, 67, 14]
orangeline = [26, 22, 83, 88, 46, 51, 24, 65, 4, 16, 49, 85, 72, 34, 25, 10, 73, 5, 59]

orange_traj_smooth = [[4.99186071539497, -1.6734691978377914],
 [5.2657461756748125, -1.5253755050737041],
 [5.631557066442651, -1.317416926803884],
 [5.968637655620625, -1.125894674134247],
 [6.276625007761082, -0.9360142330968175],
 [6.360279317314523, -0.36828382773391216],
 [6.138640057644109, -0.02250877990139919],
 [6.020420566047759, 0.37974112549064537],
 [6.042861209896298, 0.9961079641442631],
 [5.531027908835987, 1.415277796369211],
 [5.334700130044949, 1.9304991096387945],
 [5.200552464299115, 2.3718126114296845],
 [4.842658583329429, 2.365083003553543]]

green_traj_smooth = [[4.99186071539497, -1.6734691978377914],
 [5.2657461756748125, -1.5253755050737041],
 [5.631557066442651, -1.317416926803884],
 [5.968637655620625, -1.125894674134247],
 [6.276625007761082, -0.9360142330968175],
 [6.360279317314523, -0.36828382773391216],
 [6.138640057644109, -0.02250877990139919],
 [6.020420566047759, 0.37974112549064537],
 [6.042861209896298, 0.9961079641442631],
 [6.598605628653459, 1.3602252403540014],
 [7.137469522966776, 1.6823175902915413],
 [7.379109817757634, 1.0265564354023125],
 [7.581571665349905, 0.4342453681100794],
 [8.019772358659838, 0.5648714313062154],
 [8.507189724676708, 0.7163857776902893],
 [8.872212831363132, 1.2912684698279133],
 [8.669533329379263, 2.053823291933056],
 [8.680084390553933, 2.625162087967611],
 [8.925498036968452, 2.9710172215095136],
 [9.306901133430133, 3.214055057389422],
 [9.575098587750434, 3.5373007576022646],
 [9.58588833838539, 4.0209043016124895],
 [8.963022951539886, 4.895947422334167],
 [8.023769759828362, 4.778874925423389],
 [7.454562042161642, 4.7866784832228815]]

blue_traj_smooth = [[4.99186071539497, -1.6734691978377914],
 [5.2657461756748125, -1.5253755050737041],
 [5.631557066442651, -1.317416926803884],
 [5.968637655620625, -1.125894674134247],
 [6.276625007761082, -0.9360142330968175],
 [6.360279317314523, -0.36828382773391216],
 [6.138640057644109, -0.02250877990139919],
 [6.020420566047759, 0.37974112549064537],
 [6.042861209896298, 0.9961079641442631],
 [6.598605628653459, 1.3602252403540014],
 [7.137469522966776, 1.6823175902915413],
 [7.379109817757634, 1.0265564354023125],
 [7.581571665349905, 0.4342453681100794],
 [8.019772358659838, 0.5648714313062154],
 [8.507189724676708, 0.7163857776902893],
 [8.872212831363132, 1.2912684698279133],
 [9.540054051501528, 0.9759108397393635],
 [10.081766812244304, 0.7182854919567013],
 [10.572331763433553, 0.4850682459710252],
 [11.007196723691475, 0.32766574878656735],
 [11.418254700406655, 0.5592209894113417],
 [11.595911475307352, 0.7811296104272182],
 [11.647489433342704, 0.9108247267170095]]

# ----------- ??????? -----------
def timed(label):
    def deco(fn):
        def inner(*a, **kw):
            import time
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                logging.info("STAGE %s took %.3fs", label, time.time() - t0)
        return inner
    return deco


# ========== ??? & ?????????? ==========
@lru_cache(maxsize=1)
def _state():
    """
    ???? dict????
      - trans: ???? UMAP ??
      - embedding: trans.embedding_
      - embeddings_all_id_df / features_all_csn_df: ?? CSV ??
      - ppt: simpleppt ?????
    """
    logging.info("STATE build start (first hit only)")
    st = {}

    @timed("load_graphsage_output.npy")
    def _load_output():
        arr = np.load(DATA_DIR / "graphsage_output.npy")
        return pd.DataFrame(arr)

    output_df = _load_output()

    @timed("fit_umap")
    def _fit_umap(df):
        return umap.UMAP(
            n_neighbors=15,
            min_dist=1e-10,
            n_components=2,
            random_state=123,
            metric="euclidean",
            local_connectivity=1,
            verbose=UMAP_VERBOSE,
        ).fit(df)

    trans = _fit_umap(output_df)
    st["trans"] = trans
    st["embedding"] = trans.embedding_

    @timed("load_csvs")
    def _load_csvs():
        emb = pd.read_csv(DATA_DIR / "embeddings_all_id_cluster.csv")
        feats = pd.read_csv(DATA_DIR / "features_all_csn_id.csv")
        return emb, feats

    emb_df, feats_df = _load_csvs()
    st["embeddings_all_id_df"] = emb_df
    st["features_all_csn_df"] = feats_df

    @timed("build_simpleppt")
    def _build_ppt(embedding):
        return simpleppt.ppt(
            embedding,
            Nodes=100,
            seed=1,
            progress=False,
            lam=200,
            sigma=0.3,
        )

    st["ppt"] = _build_ppt(st["embedding"])

    logging.info("STATE ready")
    return st


# ========== ?? API??????????? ==========
def warm():
    """?????????????????????????????"""
    _ = _state()
    return True


def get_orginal_embed():
    return _state()["embedding"].tolist()


def tranform_new_data(data):
    trans = _state()["trans"]
    embed = trans.transform(data)
    return embed


def project_to_umap(id):
    st = _state()
    latent_df = st["embeddings_all_id_df"][st["embeddings_all_id_df"]["pat_id"] == id]
    clean_latent_df = latent_df.iloc[:, 5:]
    new_embed = tranform_new_data(clean_latent_df)
    return new_embed


def get_pat_age_egfr(pat_id):
    st = _state()
    emb_df = st["embeddings_all_id_df"]
    feats_df = st["features_all_csn_df"]

    csn_list = list(emb_df[emb_df["pat_id"] == pat_id].csn)
    ages, egfrs = [], []
    for csn in csn_list:
        age = list(feats_df[feats_df["csn"] == csn].age)[0]
        egfr = list(feats_df[feats_df["csn"] == csn]["EGFR_val"])[0]
        ages.append(age)
        egfrs.append(egfr)
    return ages, egfrs


def get_ppt_trajectory():
    ppt = _state()["ppt"]
    from_list, to_list = [], []
    for i in range(len(ppt.B)):
        for j in range(len(ppt.B[i])):
            if ppt.B[i][j] == 1:
                from_list.append(ppt.F.T[i])
                to_list.append(ppt.F.T[j])
    df_from_to = pd.DataFrame({"from": from_list, "to": to_list})
    return df_from_to


def get_four_trajectory():
    # ????????? ppt ????????????
    # ppt = _state()["ppt"]
    # green = list(map(lambda x: list(x), ppt.F.T[greenline]))
    # blue  = list(map(lambda x: list(x), ppt.F.T[blueline]))
    # orange= list(map(lambda x: list(x), ppt.F.T[orangeline]))
    green = green_traj_smooth
    blue  = blue_traj_smooth
    orange= orange_traj_smooth
    return [["green", green], ["blue", blue], ["orange", orange]]

