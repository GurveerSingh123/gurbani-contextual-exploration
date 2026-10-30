import banidb
import pandas as pd
import time

df = pd.read_csv(
    "data/processed/gurbani_full_dataset.csv"
)

unique_shabads = df["shabad_id"].dropna().unique()

metadata = []

for i, shabad_id in enumerate(unique_shabads):

    print(f"Fetching Shabad metadata {i+1}/{len(unique_shabads)}: {shabad_id}")

    try:
        shabad = banidb.shabad(int(shabad_id))

        metadata.append({
            "shabad_id": shabad.get("shabad_id"),
            "writer": shabad.get("writer", ""),
            "shabad_ang": shabad.get("ang", ""),
        })

        time.sleep(0.5)

    except Exception as e:
        print(f"ERROR {shabad_id}: {e}")

metadata_df = pd.DataFrame(metadata)

df = df.merge(
    metadata_df,
    on="shabad_id",
    how="left"
)

df.to_csv(
    "data/processed/gurbani_full_dataset_enriched.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Metadata enrichment complete.")