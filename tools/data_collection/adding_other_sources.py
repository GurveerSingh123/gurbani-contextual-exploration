import os
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"

INPUT_FILE = "data/processed/gurbani_full_dataset.csv"

OUTPUT_FILE = (
    "data/processed/"
    "gurbani_full_dataset_all_steeks.csv"
)

# Existing SS source is already in the CSV as "steek"
EXISTING_SOURCE = "ss"

# Sources we want to add
ADDITIONAL_SOURCES = [
    "bdb",
    "ft",
    "ms",
    "pss"
]


# ============================================================
# LOAD EXISTING DATASET
# ============================================================

print("Loading existing dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print(
    f"Existing records: {len(df)}"
)

print(
    f"Existing columns: {len(df.columns)}"
)


# ============================================================
# CREATE MATCHING KEY
# ============================================================

# We use shabad_id + verse_id because these identify
# the exact verse across the raw API data and CSV.

df["_key"] = (
    df["shabad_id"].astype(str).str.strip()
    + "_"
    + df["verse_id"].astype(str).str.strip()
)


# ============================================================
# PREPARE LOOKUP DICTIONARIES
# ============================================================

steek_data = {
    source: {}
    for source in ADDITIONAL_SOURCES
}


# ============================================================
# READ RAW JSON FILES
# ============================================================

print("\nReading existing raw JSON files...")
print("No API calls will be made.\n")

processed_angs = 0
skipped_angs = []

for ang_no in range(1, 1431):

    raw_file = os.path.join(
        RAW_DIR,
        f"ang_{ang_no}.json"
    )

    if not os.path.exists(raw_file):
        skipped_angs.append(
            (ang_no, "file not found")
        )
        continue

    try:

        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(
            f"Skipping Ang {ang_no}: {e}"
        )

        skipped_angs.append(
            (ang_no, str(e))
        )

        continue

    processed_angs += 1

    # --------------------------------------------------------
    # Extract verses
    # --------------------------------------------------------

    for verse in data.get("page", []):

        shabad_id = str(
            verse.get("shabad_id", "")
        ).strip()

        verse_id = str(
            verse.get("verse_id", "")
        ).strip()

        if not shabad_id or not verse_id:
            continue

        key = (
            f"{shabad_id}_{verse_id}"
        )

        # ----------------------------------------------------
        # Get Punjabi Steek data
        # ----------------------------------------------------

        steek = (
            verse
            .get("steek", {})
            .get("pu", {})
        )

        if not isinstance(steek, dict):
            continue

        # ----------------------------------------------------
        # Extract only the four additional sources
        # ----------------------------------------------------

        for source in ADDITIONAL_SOURCES:

            source_data = steek.get(
                source,
                {}
            )

            if not isinstance(source_data, dict):
                continue

            text = source_data.get(
                "unicode",
                ""
            )

            if text is None:
                text = ""

            text = str(text).strip()

            if text:
                steek_data[source][key] = text


# ============================================================
# ADD NEW COLUMNS
# ============================================================

print("\nAdding Steek columns...")

for source in ADDITIONAL_SOURCES:

    column_name = f"steek_{source}"

    # --------------------------------------------------------
    # Map raw Steek data to existing dataset
    # --------------------------------------------------------

    df[column_name] = df["_key"].map(
        steek_data[source]
    )

    # Replace NaN with empty string
    df[column_name] = (
        df[column_name]
        .fillna("")
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    available = (
        df[column_name]
        .str.strip()
        .ne("")
        .sum()
    )

    missing = len(df) - available

    print(
        f"{column_name}: "
        f"{available} available | "
        f"{missing} missing"
    )


# ============================================================
# REMOVE TEMPORARY KEY
# ============================================================

df.drop(
    columns=["_key"],
    inplace=True
)


# ============================================================
# SAVE UPDATED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("ALL STEEK SOURCES ADDED")
print("========================================")

print(
    f"Original records: {len(df)}"
)

print(
    f"Original columns + new columns: "
    f"{len(df.columns)}"
)

print("\nNew columns:")

for source in ADDITIONAL_SOURCES:
    print(
        f"  steek_{source}"
    )

print(
    f"\nRaw Angs processed: "
    f"{processed_angs}/1430"
)

print(
    f"Raw Angs skipped: "
    f"{len(skipped_angs)}"
)

if skipped_angs:

    print("\nSkipped Angs:")

    for ang, reason in skipped_angs:
        print(
            f"  Ang {ang}: {reason}"
        )

print(
    f"\nOutput file:\n"
    f"{OUTPUT_FILE}"
)

print("\n========================================")