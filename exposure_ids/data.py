from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ALIASES = {'sourceip':'src_ip', 'srcip':'src_ip', 'destinationip':'dst_ip',
           'dstip':'dst_ip', 'timestamp':'timestamp', 'label':'label',
           'flowduration':'flow_duration', 'flowid':'flow_id',
           'sourceport':'src_port', 'destinationport':'dst_port'}
META = {'src_ip', 'dst_ip', 'timestamp', 'timestamp_original', 'label', 'flow_id',
        'record_id', 'file', 'completion', 'group', 'split', 'target'}

TIMESTAMP_RULE = ('Per-file strict parse, day-first. CICIDS2017 uses a 12-hour clock without AM/PM: '
                  'hours 1-7 are treated as PM (+12h), hours 8-12 are unchanged; hour 0 or above 12 '
                  'raises. ISO8601 is accepted only for the synthetic demo.')
# (format, apply 12-hour repair)
TS_FORMATS = (('%d/%m/%Y %H:%M:%S', True), ('%d/%m/%Y %H:%M', True), ('ISO8601', False))

def canonical(name):
    key = ''.join(c for c in name.strip().lower() if c.isalnum())
    return ALIASES.get(key, key)

def read_csv_with_fallback(file, nrows=None):
    """CICIDS2017 CSVs are not all UTF-8 (e.g. 0x96 en dash in attack labels).

    Try UTF-8 first, then Windows-1252, then Latin-1; report the encoding used."""
    last = None
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(file, encoding=encoding, nrows=nrows, low_memory=False), encoding
        except UnicodeDecodeError as exc:
            last = exc
    raise ValueError(f'{file.name}: could not decode with utf-8, cp1252 or latin-1: {last}')

def parse_timestamps(raw, file_name='?'):
    """Strict timestamp parsing with an explicit, counted 12-hour repair.

    Blank timestamps become NaT (rejected later by the metadata check).
    Returns (timestamps aligned to raw.index, info dict)."""
    present = raw.notna() & raw.astype(str).str.strip().ne('')
    info = {'format': None, 'pm_shifted_rows': 0, 'missing_timestamps': int((~present).sum())}
    text = raw[present].astype(str).str.strip()
    if text.empty:
        return pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns]'), info
    for fmt, repair in TS_FORMATS:
        try:
            parsed = pd.to_datetime(text, format=fmt, errors='raise')
        except (ValueError, TypeError):
            continue
        info['format'] = fmt
        if repair:
            hours = parsed.dt.hour
            if ((hours == 0) | (hours > 12)).any():
                raise ValueError(f'{file_name}: hours outside 1-12; 12-hour clock assumption violated.')
            info['hour_counts_before_repair'] = {int(k): int(v) for k, v in hours.value_counts().sort_index().items()}
            pm = hours < 8
            parsed = parsed.where(~pm, parsed + pd.Timedelta(hours=12))
            info['pm_shifted_rows'] = int(pm.sum())
        return parsed.reindex(raw.index), info
    raise ValueError(f'{file_name}: timestamps match no single known format (examples: {text.head(3).tolist()}).')

def load_flows(folder, per_file_limit=None, dayfirst=True, floor_to_minute=False,
               drop_invalid_duration=False):
    if not dayfirst:
        raise ValueError('Month-first timestamps are not supported; CICIDS2017 files are day-first.')
    files = sorted(Path(folder).glob('*.csv'))
    if not files:
        raise ValueError(f'No CSV files in {folder}. Use demo or supply labelled flows.')
    frames, hashes, encodings, blank_rows, ts_files = [], {}, {}, {}, {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        hashes[file.name] = digest.hexdigest()
        frame, encodings[file.name] = read_csv_with_fallback(file, per_file_limit)
        frame.columns = [canonical(c) for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f'{file.name}: duplicate column names after normalization.')
        missing = {'src_ip','dst_ip','timestamp','label'} - set(frame.columns)
        if missing:
            raise ValueError(f'{file.name}: missing {sorted(missing)}. The stripped MachineLearningCSV export is insufficient; use GeneratedLabelledFlows.')
        # record_id keeps the original data-row position, assigned before any removal.
        frame['record_id'] = [f'{file.name}:{i}' for i in range(len(frame))]
        frame['file'] = file.name
        # Rows with EVERY original column empty carry no information (spreadsheet
        # leftovers). Drop only those and count them per file.
        blank = frame.drop(columns=['record_id', 'file']).isna().all(axis=1)
        blank_rows[file.name] = int(blank.sum())
        frame = frame.loc[~blank].copy()
        ts, info = parse_timestamps(frame['timestamp'], file.name)
        floored = ts.dt.floor('min')
        info['floored_rows'] = int((ts.notna() & (floored != ts)).sum()) if floor_to_minute else 0
        frame['timestamp_original'] = ts          # repaired, never floored
        frame['timestamp'] = floored if floor_to_minute else ts
        ts_files[file.name] = info
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
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
    if duration.isna().any() or (~np.isfinite(duration)).any():
        raise ValueError('Invalid Flow Duration (missing or non-finite); cannot establish causal feature availability.')
    negative = duration < 0
    rows_before_duration = len(frame)
    duration_audit = {'rule': ('Negative Flow Duration is invalid. Strict mode fails; '
                               'drop_invalid_duration removes and records exactly those rows. Zero is valid.'),
                      'drop_invalid_duration': bool(drop_invalid_duration),
                      'rows_before_filter': rows_before_duration,
                      'negative_rows': int(negative.sum()),
                      'dropped_rows': 0, 'dropped_record_ids': [], 'dropped_labels': {}, 'dropped_per_file': {}}
    if negative.any():
        if not drop_invalid_duration:
            raise ValueError(f'Invalid Flow Duration: {int(negative.sum())} negative rows; cannot establish causal '
                             'feature availability. Re-run with --drop-invalid-duration to remove and record them.')
        gone = frame.loc[negative]
        duration_audit.update(dropped_rows=int(negative.sum()),
                              dropped_record_ids=gone.record_id.tolist(),
                              dropped_labels={str(k): int(v) for k, v in gone.label.value_counts().items()},
                              dropped_per_file={str(k): int(v) for k, v in gone.file.value_counts().items()})
        frame = frame.loc[~negative].reset_index(drop=True)
        duration = duration.loc[~negative].reset_index(drop=True)
    frame['completion'] = frame['timestamp'] + pd.to_timedelta(duration, unit='us')
    frame['target'] = (~frame.label.str.upper().eq('BENIGN')).astype(int)
    features = [c for c in frame if c not in META]
    for c in features:
        frame[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    # Flow identity uses the unfloored timestamp and excludes labels.
    identity = ['src_ip','dst_ip','timestamp_original'] + features
    frame['group'] = pd.util.hash_pandas_object(frame[identity], index=False).astype(str)
    conflicts = frame.groupby('group').label.nunique()
    if (conflicts > 1).any():
        raise ValueError(f'{int((conflicts > 1).sum())} identical flow records have conflicting labels; '
                         'resolve the annotation conflict with an explicit counted rule.')
    audit = {'rows':len(frame), 'files_sha256':hashes, 'file_encodings':encodings,
             'fully_blank_rows_dropped':blank_rows,
             'timestamp_rule':TIMESTAMP_RULE, 'timestamp_files':ts_files,
             'floor_to_minute':bool(floor_to_minute),
             'flow_duration':duration_audit,
             'labels':frame.label.value_counts().to_dict(),
             'labels_by_file':{str(f):{str(k):int(v) for k,v in g.label.value_counts().items()}
                               for f,g in frame.groupby('file')},
             'features':features,
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