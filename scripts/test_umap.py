# scripts/test_umap.py
import sys, os
sys.path.append("..")

from services.get_umap import get_orginal_embed, get_four_trajectory
from services.get_df_data import get_Umap_color

def main():
    print("=== Test UMAP ===")
    try:
        embed = get_orginal_embed()
        print(f"UMAP embed shape: {len(embed)} points")
    except Exception as e:
        print(f"Failed loading embed: {e}")
        embed = []

    try:
        ages, egfrs = get_Umap_color()
        print(f"Got {len(ages)} ages, {len(egfrs)} egfrs")
    except Exception as e:
        print(f"Failed loading colors: {e}")
        ages, egfrs = [], []

    print("First 5 embed points with colors:")
    for i, (xy, age, egfr) in enumerate(zip(embed, ages, egfrs)):
        print(xy, age, egfr)
        if i >= 4:
            break

    print("Trajectories:")
    print(get_four_trajectory())

if __name__ == "__main__":
    main()

