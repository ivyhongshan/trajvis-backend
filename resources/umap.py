from flask_restful import Resource
import sys
import json
import logging

sys.path.append("..")

from services.get_umap import get_orginal_embed, get_four_trajectory, project_to_umap, get_pat_age_egfr
from services.get_df_data import get_df_all_pat, get_pat_records

class Umap(Resource):
    def get(self):
        logging.info("Umap endpoint called")
        
        try:
            embed = get_orginal_embed()
            ages, egfrs = get_Umap_color()
        except Exception as e:
            logging.error(f"Failed in Umap endpoint: {e}")
            embed, ages, egfrs = [], [], []

        embed_full = [
            [x[0], x[1], age, egfr]
            for x, age, egfr in zip(embed, ages, egfrs)
        ]

        return {
            "embed": embed_full,
            "traj": get_four_trajectory()
        }

class PatProj(Resource):
    def get(self, id):
        logging.info(f"PatProj called with id={id}")
        try:
            ages, egfrs = get_pat_age_egfr(id)
            proj = project_to_umap(id)
            embed = [
                [x[0].item(), x[1].item(), age, egfr]
                for x, age, egfr in zip(proj, ages, egfrs)
            ]
        except Exception as e:
            logging.error(f"PatProj failed for id={id}: {e}")
            embed = []

        return {
            "embed": embed
        }

