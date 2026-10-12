from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ALIASES = {'sourceip':'src_ip', 'srcip':'src_ip', 'destinationip':'dst_ip',
           'dstip':'dst_ip', 'timestamp':'timestamp', 'label':'label',
           'flowduration':'flow_duration', 'flowid':'flow_id',
           'sourceport':'src_port', 'destinationport':'dst_port'}
META = {'src_ip', 'dst_ip', 'timestamp', 'label', 'flow_id', 'record_id',
        'file', 'completion', 'group', 'split', 'target'}

def canonical(name):
    key = ''.join(c for c in name.strip().lower() if c.isalnum())
    return ALIASES.get(key, key)

def load_flows(folder, per_file_limit=None, dayfirst=True):
    files = sorted(Path(folder).glob('*.csv'))
    if not files:
        raise ValueError(f'No CSV files in {folder}. Use demo or supply labelled flows.')
    frames, hashes = [], {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        hashes[file.name] = digest.hexdigest()
        frame = pd.read_csv(file, encoding='utf-8-sig', nrows=per_file_limit, low_memory=False)
        frame.columns = [canonical(c) for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f'{file.name}: duplicate column names after normalization.')
        missing = {'src_ip','dst_ip','timestamp','label'} - set(frame.columns)
        if missing:
            raise ValueError(f'{file.name}: missing {sorted(missing)}. The stripped MachineLearningCSV export is insufficient; use GeneratedLabelledFlows.')
        frame['record_id'] = [f'{file.name}:{i}' for i in range(len(frame))]
        frame['file'] = file.name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'].astype(str).str.strip(), dayfirst=dayfirst, format='mixed', errors='coerce')
    required = ['src_ip','dst_ip','timestamp','label']
    invalid = frame[required].isna().any(axis=1)
    for c in ['src_ip','dst_ip','label']:
        invalid |= frame[c].fillna('').astype(str).str.strip().eq('')
    if invalid.any():
        raise ValueError(f'{int(invalid.sum())} rows have invalid host, time or label metadata. Resolve before training.')
    for c in ['src_ip','dst_ip','label']:
        frame[c] = frame[c].astype(str).str.strip()
    if 'flow_duration' not in frame:
        raise ValueError('Flow Duration is required to make predictions at flow completion.')
    duration = pd.to_numeric(frame['flow_duration'], errors='coerce')
    if duration.isna().any() or (~np.isfinite(duration)).any() or (duration < 0).any():
        raise ValueError('Invalid Flow Duration; cannot establish causal feature availability.')
    frame['completion'] = frame['timestamp'] + pd.to_timedelta(duration, unit='us')
    frame['target'] = (~frame.label.str.upper().eq('BENIGN')).astype(int)
    features = [c for c in frame if c not in META]
    for c in features:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    # Flow identity excludes labels so conflicting annotations cannot split apart.
    identity = ['src_ip','dst_ip','timestamp'] + features
    frame['group'] = pd.util.hash_pandas_object(frame[identity], index=False).astype(str)
    conflicts = frame.groupby('group').label.nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical flow records have conflicting labels; resolve the annotation conflict.')
    audit = {'rows':len(frame), 'files_sha256':hashes,
             'labels':frame.label.value_counts().to_dict(), 'features':features,
             'exact_duplicate_rows':int(frame.group.duplicated().sum()),
             'feature_duplicate_rows':int(frame[features].duplicated().sum()),
             'missing_values':{c:int(frame[c].isna().sum()) for c in features},
             'per_file_row_limit':per_file_limit,
             'sampling_note':'Prefix per file for development only' if per_file_limit else 'All rows loaded'}
    return frame, features, audit

def split_flows(frame, protocol, seed, gap_seconds=0):
    if protocol not in {'random','inductive','chronological'}:
        raise ValueError('Unknown protocol')
    frame = frame.copy()
    groups = frame.groupby('group', sort=True).agg(target=('target','first'), completion=('completion','min'))
    if protocol == 'chronological':
        # Timestamp ties stay together; all exact duplicates share one completion.
        ordered = groups.sort_values('completion', kind='stable')
        n = len(ordered)
        if n < 10: raise ValueError('At least 10 distinct flows required')
        cut1 = ordered.completion.iloc[int(n*.6)]
        cut2 = ordered.completion.iloc[int(n*.8)]
        if cut1 >= cut2: raise ValueError('Not enough distinct times for a chronological split')
        gap = pd.Timedelta(seconds=gap_seconds)
        groups['split'] = np.where(groups.completion < cut1-gap, 'train',
                          np.where((groups.completion >= cut1) & (groups.completion < cut2-gap), 'val',
                          np.where(groups.completion >= cut2, 'test', 'excluded')))
    else:
        rng = np.random.default_rng(seed)
        groups['split'] = 'test'
        for _, part in groups.groupby('target'):
            ids = part.index.to_numpy().copy(); rng.shuffle(ids)
            n = len(ids)
            if n < 5: raise ValueError('Each binary class needs at least five distinct records')
            groups.loc[ids[:int(n*.6)],'split'] = 'train'
            groups.loc[ids[int(n*.6):int(n*.8)],'split'] = 'val'
    frame['split'] = frame.group.map(groups.split)
    frame = frame.loc[frame.split != 'excluded'].reset_index(drop=True)
    for name in ['train','val','test']:
        if not (frame.split == name).any(): raise ValueError(f'Empty {name} partition')
    if frame.loc[frame.split == 'train','target'].nunique() != 2:
        raise ValueError('Training partition has only one binary class. Change time cutoffs/data scope; do not silently stratify time by labels.')
    return frame

def preprocess(frame, features):
    train = frame.split.eq('train').to_numpy()
    raw = frame[features].to_numpy(dtype=np.float64)
    keep = np.isfinite(raw[train]).any(axis=0)
    raw = raw[:,keep]; names = np.array(features)[keep].tolist()
    if not names: raise ValueError('No usable numeric training features')
    med = np.nanmedian(raw[train], axis=0)
    raw = np.where(np.isfinite(raw), raw, med)
    mean, std = raw[train].mean(axis=0), raw[train].std(axis=0)
    std[std < 1e-12] = 1
    x = ((raw-mean)/std).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError('Preprocessed features overflow; audit feature ranges')
    return x, {'features':names,'median':med.tolist(),'mean':mean.tolist(),'std':std.tolist(),
               'fit_partition':'train'}
