# main.py
import logging
import os
import threading
from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("startup")

# -------------------------------
# Flask 初始化
# -------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

preload_status = {"state": "not started", "error": None}

# -------------------------------
# 注册 API
# -------------------------------
from resources.patient import Patient, AllPatient
from resources.indicator import Indicator, AllIndicator
from resources.umap import Umap, PatProj
from resources.labtest import Labtest
from resources.analysis import Analysis, AnalysisDist

api_bp = Blueprint("api", __name__, url_prefix="/api")
api = Api(api_bp)

api.add_resource(Patient,      "/patient/<int:id>")
api.add_resource(AllPatient,   "/patients")
api.add_resource(AllIndicator, "/indicators")
api.add_resource(Indicator,    "/indicator/<att_name>")
api.add_resource(Umap,         "/umap")
api.add_resource(Labtest,      "/labtest/<int:id>")
api.add_resource(PatProj,      "/umap/<int:id>")
api.add_resource(Analysis,     "/analysis/<int:id>")
api.add_resource(AnalysisDist, "/analysis/dist/<concept>")
app.register_blueprint(api_bp)

# -------------------------------
# 后台预加载逻辑
# -------------------------------
def startup_preload():
    global preload_status
    preload_status["state"] = "running"
    try:
        log.info("Preloading datasets and models in background...")
        from services.get_df_data import (
            load_ckd_data_df, load_ckd_crf_demo,
            load_features_all_csn, load_acr_df_pats
        )
        from services.get_analysis import getTrajectoryPoints, get_neigh_graphsage
        from services import get_umap

        load_ckd_data_df()
        load_ckd_crf_demo()
        load_features_all_csn()
        load_acr_df_pats()
        getTrajectoryPoints()
        get_neigh_graphsage()
        get_umap.warm()   # 确保 UMAP artifacts 也加载

        preload_status["state"] = "done"
        log.info("Preload complete.")
    except Exception as e:
        preload_status["state"] = "error"
        preload_status["error"] = str(e)
        log.error(f"Preload failed: {e}")

# 只在容器主进程里跑一次
if os.getenv("PRELOAD_ONCE", "1") == "1":
    # 用一个文件锁 / 环境变量标记，防止多个 worker 重复
    preload_flag = "/tmp/preload_done"
    if not os.path.exists(preload_flag):
        threading.Thread(target=startup_preload, daemon=True).start()
        with open(preload_flag, "w") as f:
            f.write("done")

# -------------------------------
# 基础路由
# -------------------------------
@app.get("/__routes__")
def routes():
    return {"routes": sorted([str(r) for r in app.url_map.iter_rules()])}

@app.get("/health")
def health():
    return "ok", 200

@app.get("/preload-status")
def get_preload_status():
    code = 200 if preload_status["state"] != "error" else 500
    return preload_status, code

# -------------------------------
# 本地运行
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
