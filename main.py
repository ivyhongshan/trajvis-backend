# main.py
import logging
logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger("startup")

from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS

log.info("Start importing resources...")

# ??? import ?????????????
log.info("Importing resources.patient...")
from resources.patient import Patient, AllPatient
log.info("OK: resources.patient")

log.info("Importing resources.indicator...")
from resources.indicator import Indicator, AllIndicator
log.info("OK: resources.indicator")

log.info("Importing resources.umap...")
from resources.umap import Umap, PatProj
log.info("OK: resources.umap")

log.info("Importing resources.labtest...")
from resources.labtest import Labtest
log.info("OK: resources.labtest")

log.info("Importing resources.analysis...")
from resources.analysis import Analysis, AnalysisDist
log.info("OK: resources.analysis")

log.info("All resources imported successfully.")

app = Flask(__name__)

# ????
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Blueprint + API
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

# ??????
@app.get("/__routes__")
def routes():
    return {"routes": sorted([str(r) for r in app.url_map.iter_rules()])}

# ????
@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    # ??????Cloud Run ? gunicorn ??
    app.run(host="0.0.0.0", port=8080, debug=False)s
