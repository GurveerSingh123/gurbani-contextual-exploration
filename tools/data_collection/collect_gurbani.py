import banidb
import pandas as pd
import json
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

# Sri Guru Granth Sahib Ji has 1430 Angs
START_ANG = 1
END_ANG = 1430

# Selected Punjabi Steek source
STEEK_SOURCE = "ss"

# Source ID for Sri Guru Granth Sahib Ji
SOURCE_ID = "G"

# Output directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# FUNCTION: GET STEEK
# ============================================================

def get_steek(verse, source="ss"):
    """
    Extract Punjabi Unicode Steek from the selected source.
    """

    try:
        steek = verse["steek"]["pu"][source]["unicode"]

        if steek is None:
            return ""

        return steek.strip()

    except (KeyError, TypeError):
        return ""


# ============================================================
# FUNCTION: CHECK STRUCTURAL VERSE
# ============================================================

def is_structural_line(verse_text):
    """
    Identify common structural/header lines that should not
    be treated as semantic Pankatis.
    """

    if not verse_text:
        return True

    text = verse_text.strip()

    structural_terms = [
        "ਸਲੋਕੁ ॥",
        "ਸਲੋਕ ॥",
        "ਪਉੜੀ ॥",
        "ਰਹਾਉ ॥",
        "ਰਹਾਉ।",
        "ਅਸਟਪਦੀ ॥",
        "ਅਸਟਪਦੀ",
        "ਛੰਤੁ ॥",
        "ਵਾਰ ॥"
    ]

    return text in structural_terms


# ============================================================
# STEP 1: COLLECT MISSING RAW DATA
# ============================================================

failed_angs = []

print("\n========================================")
print("STEP 1: RAW DATA COLLECTION")
print("========================================")

for ang_no in range(START_ANG, END_ANG + 1):

    raw_file = os.path.join(
        RAW_DIR,
        f"ang_{ang_no}.json"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # If raw file already exists, DO NOT download again.
    # --------------------------------------------------------

    if os.path.exists(raw_file):

        print(
            f"Ang {ang_no}: already exists - skipping download"
        )

        continue

    print(
        f"Collecting Ang {ang_no}/{END_ANG}..."
    )

    try:

        # ----------------------------------------------------
        # Get complete Ang
         # ----------------------------------------------------

        page_data = banidb.angs(
            ang_no,
            source_id=SOURCE_ID,
            larivaar=False,
            steek=True,
            translit=False
        )

        # ----------------------------------------------------
        # Save original API response
        # ----------------------------------------------------

        with open(
            raw_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                page_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"  Saved Ang {ang_no}"
        )

        time.sleep(0.5)

    except Exception as e:

        print(
            f"ERROR collecting Ang {ang_no}: {e}"
        )

        failed_angs.append({
            "ang": ang_no,
            "error": str(e)
        })


# ============================================================
# STEP 2: PROCESS ALL RAW FILES
# ============================================================

print("\n========================================")
print("STEP 2: PROCESSING ALL RAW FILES")
print("========================================")

records = []

missing_raw_angs = []


for ang_no in range(START_ANG, END_ANG + 1):

    raw_file = os.path.join(
        RAW_DIR,
        f"ang_{ang_no}.json"
    )

    # --------------------------------------------------------
    # Check whether raw file exists
    # --------------------------------------------------------

    if not os.path.exists(raw_file):

        print(
            f"WARNING: Raw file missing for Ang {ang_no}"
        )

        missing_raw_angs.append(ang_no)

        continue

    print(
        f"Processing Ang {ang_no}/{END_ANG}..."
    )

    try:

        # ----------------------------------------------------
        # Load raw JSON
        # ----------------------------------------------------

        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as f:

            page_data = json.load(f)

        # ----------------------------------------------------
        # Source information
        # ----------------------------------------------------

        source_info = page_data.get(
            "source",
            {}
        )

        source = source_info.get(
            "english",
            "Sri Guru Granth Sahib Ji"
        )

        ang = source_info.get(
            "ang_no",
            ang_no
        )

        # ----------------------------------------------------
        # Extract verses
        # ----------------------------------------------------

        verses = page_data.get(
            "page",
            []
        )

        for verse in verses:

            verse_id = verse.get(
                "verse_id",
                ""
            )

            shabad_id = verse.get(
                "shabad_id",
                ""
            )

            pankati = verse.get(
                "verse",
                ""
            ).strip()

            # Skip empty verses
            if not pankati:
                continue

            # ------------------------------------------------
            # Structural check
            # ------------------------------------------------

            structural = is_structural_line(
                pankati
            )

            # ------------------------------------------------
            # Extract Steek
            # ------------------------------------------------

            steek = get_steek(
                verse,
                STEEK_SOURCE
            )

            # ------------------------------------------------
            # Create semantic text
            # ------------------------------------------------

            semantic_text = (
                f"{pankati} {steek}"
            ).strip()

            # ------------------------------------------------
            # Add record
            # ------------------------------------------------

            records.append({

                "id": len(records) + 1,

                "ang": ang,

                "shabad_id": shabad_id,

                "verse_id": verse_id,

                "pankati": pankati,

                "steek": steek,

                "semantic_text": semantic_text,

                "writer": "",

                "source": source,

                "steek_source": STEEK_SOURCE,

                "is_structural": structural
            })

    except Exception as e:

        print(
            f"ERROR processing Ang {ang_no}: {e}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

output_file = os.path.join(
    PROCESSED_DIR,
    "gurbani_full_dataset.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CREATE SEMANTIC DATASET
# ============================================================

semantic_df = df[
    df["is_structural"] == False
].copy()

semantic_file = os.path.join(
    PROCESSED_DIR,
    "gurbani_semantic_dataset.csv"
)

semantic_df.to_csv(
    semantic_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MISSING STEEK ANALYSIS
# ============================================================

missing_steek = (
    df["steek"]
    .fillna("")
    .str.strip()
    .eq("")
).sum()

total = len(df)

if total > 0:

    missing_percentage = (
        missing_steek / total * 100
    )

else:

    missing_percentage = 0


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df[
    df.duplicated(
        subset=[
            "shabad_id",
            "verse_id"
        ],
        keep=False
    )
]


# ============================================================
# STRUCTURAL STATISTICS
# ============================================================

structural_count = (
    df["is_structural"] == True
).sum()

semantic_count = (
    df["is_structural"] == False
).sum()


# ============================================================
# SAVE FAILED / MISSING ANGS
# ============================================================

all_problem_angs = []

for item in failed_angs:

    all_problem_angs.append(item)

for ang in missing_raw_angs:

    all_problem_angs.append({
        "ang": ang,
        "error": "Raw JSON file missing"
    })


failed_file = os.path.join(
    PROCESSED_DIR,
    "failed_angs.json"
)

with open(
    failed_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_problem_angs,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("========================================")
print("FULL DATASET PROCESSING COMPLETE")
print("========================================")

print(
    f"Angs requested: "
    f"{END_ANG - START_ANG + 1}"
)

print(
    f"Raw Angs available: "
    f"{END_ANG - len(missing_raw_angs)}"
)

print(
    f"Missing raw Angs: "
    f"{len(missing_raw_angs)}"
)

print(
    f"Total records: "
    f"{total}"
)

print(
    f"Semantic records: "
    f"{semantic_count}"
)

print(
    f"Structural records: "
    f"{structural_count}"
)

print(
    f"Missing Steek: "
    f"{missing_steek}/{total} "
    f"({missing_percentage:.2f}%)"
)

print(
    f"Duplicate verse records: "
    f"{len(duplicates)}"
)

print(
    f"\nFull dataset saved at:\n"
    f"{output_file}"
)

print(
    f"\nSemantic dataset saved at:\n"
    f"{semantic_file}"
)

print(
    f"\nProblem Angs saved at:\n"
    f"{failed_file}"
)

print("\n========================================")