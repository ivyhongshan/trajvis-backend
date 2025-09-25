# tests/test_get_analysis.py
import sys
import pprint
import os

# 把项目根目录加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sys.path.append("..")  # 确保能找到 services 包
from services import get_analysis as ga


def run_all_tests():
    pp = pprint.PrettyPrinter(indent=2)

    print("\n=== Test getTrajectoryPoints ===")
    traj_points = ga.getTrajectoryPoints()
    pp.pprint({k: v[:3] for k, v in traj_points.items()})  # 打印前三个点

    print("\n=== Test get_pat_sex_distribution ===")
    pp.pprint(ga.get_pat_sex_distribution())

    print("\n=== Test get_pat_race_distribution ===")
    pp.pprint(ga.get_pat_race_distribution())

    print("\n=== Test get_concept_distribution('lab_hb') ===")
    try:
        x, y = ga.get_concept_distribution("lab_hb")
        print("x sample:", x[:5])
        print("y sample:", y[0][:2])
    except Exception as e:
        print("❌ get_concept_distribution failed:", e)

    print("\n=== Test fakeBlueData / fakegreenData / fakeOrangeData ===")
    xs = [40, 50, 60]
    widths = [10, 15, 20]
    print("Blue:", ga.fakeBlueData(xs, widths)[:2])
    print("Green:", ga.fakegreenData(xs, widths)[:2])
    print("Orange:", ga.fakeOrangeData(xs, widths)[:2])

    print("\n=== Test get_one_pat_cluster_label_by_age ===")
    try:
        age_last, ages, gw, bw, ow = ga.get_one_pat_cluster_label_by_age(1)
        print("age_last:", age_last)
        print("ages[:5]:", ages[:5])
        print("green_width[:5]:", gw[:5])
    except Exception as e:
        print("❌ get_one_pat_cluster_label_by_age failed:", e)

    print("\n=== Test get_three_uncertainty ===")
    try:
        ages, gw, bw, ow = ga.get_three_uncertainty(1)
        print("ages[:5]:", ages[:5])
        print("green_width[:5]:", gw[:5])
    except Exception as e:
        print("❌ get_three_uncertainty failed:", e)

    print("\n=== Test getThreePossibility ===")
    ages = [30, 31, 32, 33, 34]
    gw = [20, 21, 19, 18, 20]
    bw = [10, 11, 12, 13, 14]
    ow = [15, 14, 16, 15, 17]
    ages_new, gw_new, bw_new, ow_new = ga.getThreePossibility(ages, gw, bw, ow)
    print("ages_new[:5]:", ages_new[:5])
    print("green_width[:5]:", gw_new[:5])


if __name__ == "__main__":
    run_all_tests()
