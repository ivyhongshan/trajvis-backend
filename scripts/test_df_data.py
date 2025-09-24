import sys
from pathlib import Path

# ????? services ??
sys.path.append(str(Path(__file__).resolve().parents[1]))

from services import get_df_data

def main():
    print("? ?? get_df_data ????????")
    required_funcs = [
        "get_pat_records",
        "get_df_all_pat",
        "get_pat_demo",
        "get_pat_unique_concept",
        "get_pat_kidney_risk",
        "get_profile_date"
    ]

    for func in required_funcs:
        if hasattr(get_df_data, func):
            print(f" - {func}: ??")
        else:
            print(f" - {func}: ? ??")

    # ?????????
    try:
        pats = get_df_data.get_df_all_pat()
        print(f"??? {len(pats)} ???ID, ??: {pats[:5]}")
        if pats:
            pat_id = pats[0]
            records = get_df_data.get_pat_records(pat_id)
            print(f"?? {pat_id} ????: {len(records)}")
    except Exception as e:
        print("?? ??????:", e)

if __name__ == "__main__":
    main()

