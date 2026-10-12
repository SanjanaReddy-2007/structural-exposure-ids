from pathlib import Path
import numpy as np
import pandas as pd

def make_demo(folder,rows=1200,seed=42):
    """Synthetic traffic for pipeline checks only; never a research dataset."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    # Recurrent bursts ensure both classes occur across chronological partitions.
    attack=(np.arange(rows)//30%4==1).astype(int)
    src=[f'10.0.0.{1+int((i//12)%24)}' for i in range(rows)]
    dst=[f'10.1.0.{1+int(rng.integers(8))}' for _ in range(rows)]
    frame=pd.DataFrame({'Source IP':src,'Destination IP':dst,
        'Timestamp':pd.date_range('2017-07-04 09:00:00',periods=rows,freq='s').astype(str),
        'Flow Duration':rng.integers(1000,900000,rows),
        'Destination Port':rng.choice([22,80,443],rows),
        'Total Fwd Packets':rng.poisson(10+attack*3,rows),
        'Flow Bytes/s':rng.lognormal(5+attack*.25,.9,rows),
        'Flow IAT Mean':rng.normal(100+attack*10,25,rows),
        'Label':np.where(attack,'SYNTHETIC_ATTACK','BENIGN')})
    path=folder/'synthetic.csv'
    if path.exists():raise ValueError(f'{path} already exists; choose a fresh demo folder')
    frame.to_csv(path,index=False)
    (folder/'SYNTHETIC_ONLY.txt').write_text('Synthetic pipeline validation only. Not CICIDS2017. Do not report scores as research findings.\n')
    return path
