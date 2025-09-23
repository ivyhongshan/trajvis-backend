import os
import importlib
from flask import Flask, Blueprint
from flask_restful import Api
from flask_cors import CORS

from resources.patient import Patient, AllPatient
from resources.indicator import Indicator, AllIndicator
from resources.umap import Umap, PatProj
from resources.labtest import Labtest
from resources.analysis import Analysis, AnalysisDist

app = Flask(__name__)

@app.get("/")
def index():
    return "trajvis backend is running"

@app.get("/health")
def health():
    return {"status": "ok"}

# --- CORS ?? ---
# ?????? URL ???????? FE_URL?????? "*" ???
FE_URL = os.getenv("FE_URL", "*")
CORS(
    app,
    resources={r"/*": {"origins": FE_URL}},
    supports_credentials=False,
    allow_headers=["*"],
    methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"]
)

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

api.add_resource(Patient, '/api/patient/<int:id>')
api.add_resource(AllPatient, '/api/patients')
api.add_resource(AllIndicator, '/api/indicators')
api.add_resource(Indicator, '/api/indicator/<att_name>')
api.add_resource(Umap, '/api/umap')
api.add_resource(Labtest, '/api/labtest/<int:id>')
api.add_resource(PatProj, '/api/umap/<int:id>')
api.add_resource(Analysis, '/api/analysis/<int:id>')
api.add_resource(AnalysisDist, '/api/analysis/dist/<concept>')

app.register_blueprint(api_bp)

if __name__ == '__main__':
    # Cloud Run ???? 8080?????? PORT ????
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=PORT, debug=False)

