"""Read-only diagnostic: why does load_flows reject Flow Duration?

Counts, per CSV, blank rows and rows whose Flow Duration is empty, non-numeric,
infinite, negative or zero, and shows the label mix and a few example rows.
Changes nothing. Usage (from the repo root):
    python scripts/diagnose_flow_duration.py data/raw/GeneratedLabelledFlows
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exposure_ids.data import canonical  # noqa: E402

WANT = {'flow_id', 'src_ip', 'dst_ip', 'timestamp', 'flow_duration', 'label'}

def read(file):
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(file, encoding=enc, low_memory=False,
                               usecols=lambda c: canonical(c) in WANT)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f'cannot decode {file}')

def main(folder):
    files = sorted(Path(folder).glob('*.csv'))
    if not files:
        raise SystemExit(f'No CSV files in {folder}')
    total = {}
    for f in files:
        df = read(f)
        df.columns = [canonical(c) for c in df.columns]
        n = len(df)
        blank = df.isna().all(axis=1)
        df = df.loc[~blank]
        raw = df['flow_duration']
        num = pd.to_numeric(raw, errors='coerce')
        empty = raw.isna()
        nonnum = num.isna() & ~empty
        inf = np.isinf(num.fillna(0))
        neg = num < 0
        zero = num == 0
        bad = empty | nonnum | inf | neg
        print(f'\n== {f.name}: {n} rows, {int(blank.sum())} fully blank, {len(df)} real flows')
        print(f'   duration empty={int(empty.sum())} non-numeric={int(nonnum.sum())} '
              f'inf={int(inf.sum())} negative={int(neg.sum())} zero={int(zero.sum())} '
              f'-> invalid total={int(bad.sum())}')
        if neg.any():
            print(f'   negative min={num[neg].min():g}, max={num[neg].max():g}')
        if bad.any():
            print('   labels of invalid rows:', df.loc[bad, 'label'].value_counts().to_dict())
            show = df.loc[bad, [c for c in ['flow_id', 'timestamp', 'flow_duration', 'label'] if c in df]].head(3)
            print('   examples:\n' + show.to_string(index=False))
        for k, v in dict(rows=len(df), invalid=int(bad.sum()), negative=int(neg.sum()),
                         empty=int(empty.sum()), nonnumeric=int(nonnum.sum()),
                         inf=int(inf.sum())).items():
            total[k] = total.get(k, 0) + v
    print('\n== TOTAL', total)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/raw/GeneratedLabelledFlows')