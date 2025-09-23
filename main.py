# main.py
from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS

from resources.patient import Patient, AllPatient
from resources.indicator import Indicator, AllIndicator
from resources.umap import Umap, PatProj
from resources.labtest import Labtest
from resources.analysis import Analysis, AnalysisDist

app = Flask(__name__)

# ??? /api/* ? CORS????????? origins ?????????
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ???? /api ?? Blueprint ? url_prefix ?
api_bp = Blueprint("api", __name__, url_prefix="/api")
api = Api(api_bp)

# ??????????? /api/ ??
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

# ????
@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    # ?????Cloud Run ?? gunicorn ????????
    app.run(host="0.0.0.0", port=8080, debug=False)

