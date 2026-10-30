import pandas as pd
import re

# ============================================================
# FILE PATHS
# ============================================================

GURBANI_CSV = "data/processed/gurbani_full_dataset_enriched.csv"
RAAG_CSV = "data/metadata/raag_metadata.csv"
OUTPUT_CSV = "data/processed/gurbani_enriched_with_raag_id.csv"


# ============================================================
# LOAD CSV FILES
# ============================================================

gurbani = pd.read_csv(GURBANI_CSV)
raag = pd.read_csv(RAAG_CSV)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "ang" not in gurbani.columns:
    raise ValueError("Main Gurbani CSV does not contain an 'ang' column.")

required_raag_columns = ["raag_id", "angs"]

for column in required_raag_columns:
    if column not in raag.columns:
        raise ValueError(
            f"Raag metadata CSV does not contain required column: '{column}'"
        )


# ============================================================
# FUNCTION TO EXTRACT ANG RANGE
# ============================================================

def parse_ang_range(value):
    """
    Converts different Ang formats into:
        (start_ang, end_ang)

    Examples:
        984-988       -> (984, 988)
        984–988       -> (984, 988)
        [984, 988]    -> (984, 988)
        984,988       -> (984, 988)
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Find all numbers
    numbers = re.findall(r"\d+", value)

    if len(numbers) < 2:
        return None

    start_ang = int(numbers[0])
    end_ang = int(numbers[1])

    return start_ang, end_ang


# ============================================================
# BUILD ANG → RAAG ID MAPPING
# ============================================================

ang_to_raag = {}

for _, row in raag.iterrows():

    raag_id = row["raag_id"]
    angs = row["angs"]

    parsed = parse_ang_range(angs)

    if parsed is None:
        print(
            f"WARNING: Could not parse Ang range for Raag {raag_id}: {angs}"
        )
        continue

    start_ang, end_ang = parsed

    print(
        f"Raag {raag_id}: Ang {start_ang} → {end_ang}"
    )

    for ang in range(start_ang, end_ang + 1):

        if ang in ang_to_raag:
            print(
                f"WARNING: Ang {ang} has multiple Raag assignments: "
                f"{ang_to_raag[ang]} and {raag_id}"
            )

        ang_to_raag[ang] = raag_id


# ============================================================
# ASSIGN RAAG ID TO GURBANI DATASET
# ============================================================

def get_raag_id(ang):

    if pd.isna(ang):
        return pd.NA

    try:
        ang = int(float(ang))
    except (ValueError, TypeError):
        return pd.NA

    return ang_to_raag.get(ang, pd.NA)


gurbani["raag_id"] = gurbani["ang"].apply(get_raag_id)


# ============================================================
# VALIDATION
# ============================================================

print("\n========================================")
print("RAAG ASSIGNMENT SUMMARY")
print("========================================")

print(
    gurbani["raag_id"]
    .value_counts(dropna=False)
    .sort_index()
)

total_rows = len(gurbani)
assigned_rows = gurbani["raag_id"].notna().sum()
missing_rows = gurbani["raag_id"].isna().sum()

print("\nTotal Gurbani rows:", total_rows)
print("Rows assigned a Raag:", assigned_rows)
print("Rows without Raag:", missing_rows)


# ============================================================
# SHOW UNMATCHED ANGS
# ============================================================

if missing_rows > 0:

    unmatched = (
        gurbani.loc[gurbani["raag_id"].isna(), "ang"]
        .dropna()
        .unique()
    )

    print("\nUnmatched Angs:")
    print(sorted(unmatched))


# ============================================================
# SAVE
# ============================================================

gurbani.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n========================================")
print("DONE")
print("========================================")
print(f"Output saved to: {OUTPUT_CSV}")