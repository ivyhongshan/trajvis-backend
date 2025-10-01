from flask_restful import Resource
import sys
import json
import logging
import time

sys.path.append("..")

from services.get_umap import get_orginal_embed, get_four_trajectory, project_to_umap, get_pat_age_egfr
from services.get_df_data import get_df_all_pat, get_pat_records, get_Umap_color


class Umap(Resource):
    def get(self):
        logging.info("Umap endpoint called")
        t0 = time.time()
        
        try:
            embed = get_orginal_embed()
            logging.info(f"get_orginal_embed took {time.time()-t0:.2f}s")
            
            ages, egfrs = get_Umap_color()
            logging.info(f"get_Umap_color took {time.time()-t0:.2f}s total")
        except Exception as e:
            logging.error(f"Failed in Umap endpoint: {e}")
            embed, ages, egfrs = [], [], []

        embed_full = [
            [x[0], x[1], age, egfr]
            for x, age, egfr in zip(embed, ages, egfrs)
        ]

        total = time.time() - t0
        logging.info(f"/api/umap total time {total:.2f}s")

        return {
            "embed": embed_full,
            "traj": get_four_trajectory(),
            "time": total
        }


class PatProj(Resource):
    def get(self, id):
        logging.info(f"PatProj called with id={id}")
        t0 = time.time()
        try:
            ages, egfrs = get_pat_age_egfr(id)
            logging.info(f"get_pat_age_egfr took {time.time()-t0:.2f}s")

            proj = project_to_umap(id)
            logging.info(f"project_to_umap took {time.time()-t0:.2f}s total")

            embed = [
                [x[0].item(), x[1].item(), age, egfr]
                for x, age, egfr in zip(proj, ages, egfrs)
            ]
        except Exception as e:
            logging.error(f"PatProj failed for id={id}: {e}")
            embed = []

        total = time.time() - t0
        logging.info(f"/api/umap/<id> total time {total:.2f}s")

        return {
            "embed": embed,
            "time": total
        }
