import requests
import pandas as pd
import json
import os


API_BASE = "https://api.banidb.com/v2"

RAAGS_URL = f"{API_BASE}/raags"
WRITERS_URL = f"{API_BASE}/writers"

OUTPUT_FILE = "data/processed/raag_metadata.csv"


# ============================================================
# FETCH RAAGS
# ============================================================

print("Fetching Raag metadata...")

response = requests.get(
    RAAGS_URL,
    timeout=30
)

response.raise_for_status()

raag_json = response.json()

rows = raag_json.get("rows", [])

print(f"Raag rows received: {len(rows)}")


# ============================================================
# FETCH WRITERS ONCE
# ============================================================

print("Fetching writer metadata...")

response = requests.get(
    WRITERS_URL,
    timeout=30
)

response.raise_for_status()

writer_json = response.json()

writer_rows = writer_json.get("rows", [])

writer_lookup = {}

for writer_row in writer_rows:

    writer_id = writer_row.get("WriterID")

    writer_lookup[writer_id] = writer_row.get(
        "WriterEnglish",
        ""
    )

print(
    f"Writers loaded: {len(writer_lookup)}"
)


# ============================================================
# EXTRACT RAAGS
# ============================================================

metadata = []

for row in rows[1:]:

    # --------------------------------------------------------
    # Writers
    # --------------------------------------------------------

    writer_ids = row.get("Writers") or []

    writer_names = [
        writer_lookup.get(writer_id, "")
        for writer_id in writer_ids
    ]

    writer_names = [
        name for name in writer_names
        if name
    ]

    # --------------------------------------------------------
    # Sargun
    # --------------------------------------------------------

    sargun = row.get("Sargun") or {}

    # --------------------------------------------------------
    # Ang ranges for Sri Guru Granth Sahib Ji
    # --------------------------------------------------------

    source_info = row.get("SourceInfo") or []

    ang_ranges = {}

    for info in source_info:

        if info.get("SourceID") == "G":

            ang_ranges["G"] = [
                info.get("Start"),
                info.get("End")
            ]

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    metadata.append({

        "raag_id":
            row.get("RaagID", ""),

        "raag_uni":
            row.get("RaagUnicode", ""),

        "raag":
            row.get("RaagEnglish", ""),

        "time_of_raag":
            json.dumps([
                row.get("StartTime"),
                row.get("EndTime")
            ], ensure_ascii=False),

        "common_themes":
            row.get("CommonThemes", ""),

        "feeling":
            row.get("Feeling", ""),

        "angs":
            json.dumps(
                ang_ranges,
                ensure_ascii=False
            ),

        "overview":
            row.get("Overview", ""),

        "advanced":
            row.get("Advanced", ""),

        "musical_composure":
            row.get("MusicalComposure", ""),

        "placement":
            row.get("Placement", ""),

        "season":
            row.get("Season", ""),

        "aroh":
            sargun.get("Aroh", ""),

        "avroh":
            sargun.get("Avroh", ""),

        "vadi":
            sargun.get("Vadi", ""),

        "samvadi":
            sargun.get("Samvadi", ""),

        "sur":
            sargun.get("Sur", ""),

        "thaat":
            sargun.get("Thaat", ""),

        "jaati":
            row.get("Jaati", ""),

        "writers":
            json.dumps(
                writer_names,
                ensure_ascii=False
            )
    })


# ============================================================
# SAVE
# ============================================================

metadata_df = pd.DataFrame(metadata)

os.makedirs(
    "data/processed",
    exist_ok=True
)

metadata_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("RAAG METADATA COLLECTION COMPLETE")
print("========================================")

print(
    f"Raags collected: {len(metadata_df)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print(
    metadata_df[
        [
            "raag_id",
            "raag_uni",
            "raag",
            "angs"
        ]
    ].to_string(index=False)
)