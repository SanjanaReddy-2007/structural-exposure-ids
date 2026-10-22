import argparse
import glob
import os
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
a = p.parse_args()

for f in sorted(glob.glob(os.path.join(a.data, "*.csv"))):
    df = pd.read_csv(f, encoding="cp1252", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" not in df.columns:
        print(f, "-> no Timestamp column; columns:", list(df.columns)[:10])
        continue
    raw = df["Timestamp"].dropna().astype(str)
    print("=" * 70)
    print(os.path.basename(f), "| rows:", len(df), "| non-null ts:", len(raw))
    print("first 5 raw:", raw.head(5).tolist())
    print("last 5 raw:", raw.tail(5).tolist())
    print("has seconds (2 colons):", (raw.str.count(":") == 2).mean().round(4))
    print("has AM/PM:", raw.str.contains("AM|PM", case=False).mean().round(4))
    t = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    print("unparsed:", int(t.isna().sum()))
    t = t.dropna()
    if len(t):
        print("min:", t.min(), "| max:", t.max())
        print("hour counts:", t.dt.hour.value_counts().sort_index().to_dict())
        print("distinct seconds values:", t.dt.second.nunique())
    print("is raw order monotonic:", bool(t.is_monotonic_increasing))