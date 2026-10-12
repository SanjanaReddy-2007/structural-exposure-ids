import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_flows, split_flows, preprocess
from .graph import build_graph, partition_edges, retain_cross_edges, exposure, rewire, reduce_cross_degree_preserving

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def prepare(args):
    output=Path(args.out)
    if output.exists() and any(output.iterdir()):raise ValueError('Output folder is nonempty; choose a fresh folder to preserve run provenance.')
    output.mkdir(parents=True,exist_ok=True)
    frame,features,audit=load_flows(args.data,args.per_file_limit,not args.monthfirst)
    frame=split_flows(frame,args.protocol,args.split_seed,args.gap_seconds)
    x,processor=preprocess(frame,features)
    splits=frame.split.to_numpy()
    audit['partition_labels']={name:part.label.value_counts().to_dict() for name,part in frame.groupby('split')}
    audit['synthetic_only']=(Path(args.data)/'SYNTHETIC_ONLY.txt').exists()
    audit['unseen_test_labels']=sorted(set(frame.loc[frame.split=='test','label'])-set(frame.loc[frame.split=='train','label']))
    fingerprints=pd.util.hash_pandas_object(frame[features],index=False)
    train_fp=set(fingerprints[frame.split=='train'])
    audit['test_feature_rows_also_in_training']=int(fingerprints[frame.split=='test'].isin(train_fp).sum())
    frame[['record_id','group','src_ip','dst_ip','completion','label','target','split']].to_csv(output/'split_manifest.csv',index=False)
    write_json(output/'audit.json',audit);write_json(output/'preprocessing.json',processor)
    config={k:v for k,v in vars(args).items() if k!='func'}
    config['python']=sys.version;config['platform']=platform.platform()
    config['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(output/'config.json',config)
    # Build within partitions for inductive evaluation, including the cap selection.
    if args.protocol=='inductive':
        parts=[]
        for name in ['train','val','test']:
            ids=np.flatnonzero(splits==name)
            local=build_graph(frame.iloc[ids].reset_index(drop=True),args.window,args.neighbors)
            if len(local):parts.append(ids[local])
        full=np.concatenate(parts) if parts else np.empty((0,2),dtype=np.int64)
    elif args.protocol=='random':
        views=[]
        for heldout in ['test','val']:
            ids=np.flatnonzero(np.isin(splits,['train',heldout]))
            local=build_graph(frame.iloc[ids].reset_index(drop=True),args.window,args.neighbors)
            views.append(ids[local] if len(local) else np.empty((0,2),dtype=np.int64))
        base,val=views
        full=None
    else:
        full=build_graph(frame,args.window,args.neighbors,args.protocol=='chronological')
    if full is not None:
        base=partition_edges(full,splits,args.protocol)
        val=partition_edges(full,splits,args.protocol,'val')
    np.savez_compressed(output/'prepared.npz',x=x,y=frame.target.to_numpy(),splits=splits,edges=base,val_edges=val)
    write_json(output/'exposure.json',exposure(frame,base))
    return output,frame,x,base,val,audit

def audit_command(args):
    out,frame,x,base,val,audit=prepare(args)
    print(f'Prepared {len(frame)} flows, {x.shape[1]} features and {len(base)} message arcs in {out}')
    if audit['synthetic_only']:print('SYNTHETIC VALIDATION ONLY')

def experiment(args):
    from .models import train_neural,train_rf,predict,metrics
    import sklearn,torch,joblib
    start=time.time()
    out,frame,x,base,val,audit=prepare(args)
    write_json(out/'versions.json',{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__})
    splits=frame.split.to_numpy();y=frame.target.to_numpy();test=splits=='test'
    rows=[];predictions=[]
    levels=args.levels if args.protocol=='random' else [1.0]
    for seed in args.seeds:
        rf=train_rf(x,y,splits,seed)
        joblib.dump(rf,out/f'random_forest_{seed}.joblib')
        prob=rf.predict_proba(x)[:,list(rf.classes_).index(1)]
        rows.append({'seed':seed,'model':'random_forest','topology':'none','retention':None,**metrics(y[test],prob[test])})
        predictions.append(pd.DataFrame({'record_id':frame.loc[test,'record_id'],'target':y[test],
                    'probability_attack':prob[test],'seed':seed,'condition':f'random_forest_{seed}'}))
        mlp,details=train_neural(x,y,splits,base,val,seed,'mlp',args.hidden,args.epochs,args.patience)
        write_json(out/f'mlp_{seed}.json',details)
        torch.save(mlp.state_dict(),out/f'mlp_{seed}.pt')
        prob=predict(mlp,x,base);mlp_score=metrics(y[test],prob[test])
        rows.append({'seed':seed,'model':'mlp','topology':'none','retention':None,**mlp_score})
        predictions.append(pd.DataFrame({'record_id':frame.loc[test,'record_id'],'target':y[test],
                    'probability_attack':prob[test],'seed':seed,'condition':f'mlp_{seed}'}))
        controls=['sage_no_neighbors']+['sage']*len(levels)
        for position,kind in enumerate(controls):
            fraction=1.0 if position==0 else levels[position-1]
            edges=retain_cross_edges(base,splits,fraction,args.graph_seed)
            control_info={}
            if args.degree_controlled and kind=='sage':
                edges,control_info=reduce_cross_degree_preserving(base,frame,fraction,args.window,args.graph_seed)
                if not control_info['target_reached']:
                    print(f'Degree control at {fraction:g}: target not reached; achieved '
                          f'{control_info["achieved_cross_edges"]} cross edges versus '
                          f'{control_info["target_cross_edges"]} requested.',flush=True)
            for topology in (['original','rewired'] if kind=='sage' and args.rewire else ['original']):
                used=edges;swap_info={}
                if topology=='rewired':used,swap_info=rewire(edges,frame,args.window,args.protocol=='chronological',seed)
                tag=f'{kind}_{seed}_{fraction:g}_{topology}'
                model,details=train_neural(x,y,splits,used,val,seed,kind,args.hidden,args.epochs,args.patience)
                torch.save(model.state_dict(),out/f'{tag}.pt')
                write_json(out/f'{tag}.json',{**details,'rewiring':swap_info,'connectivity_control':control_info})
                np.save(out/f'{tag}_edges.npy',used)
                prob=predict(model,x,used);score=metrics(y[test],prob[test]);exp=exposure(frame,used)
                rows.append({'seed':seed,'model':kind,'topology':topology,'retention':fraction,
                             'gap_to_mlp':score['macro_f1']-mlp_score['macro_f1'],
                             'connectivity_method':'degree_controlled' if args.degree_controlled and kind=='sage' else 'deletion',
                             'connectivity_target_reached':control_info.get('target_reached'),
                             'rewiring_swaps':swap_info.get('accepted_swaps'),**score,**exp})
                predictions.append(pd.DataFrame({'record_id':frame.loc[test,'record_id'],'target':y[test],
                    'probability_attack':prob[test],'seed':seed,'condition':tag}))
                print(f'{tag}: macro-F1 {score["macro_f1"]:.4f}; exposure {exp["two_hop_test_exposure"]:.3f}',flush=True)
    write_json(out/'results.json',{'synthetic_only':audit['synthetic_only'],'elapsed_seconds':time.time()-start,'runs':rows,
        'limits':['Full-batch CPU prototype; use manageable subsets.',
                  'Fixed hyperparameters; no equal-budget tuning study yet.',
                  'Deletion sweep changes degree; not an isolated causal overlap effect.',
                  'Rewiring preserves degree and partition pair counts, not K-hop exposure.',
                  'Seeds summarize optimization variability, not population confidence intervals.']})
    table=pd.DataFrame(rows);table.to_csv(out/'results.csv',index=False)
    table.groupby(['model','topology','retention'],dropna=False).agg(
        runs=('seed','count'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_seed_sd=('macro_f1','std'),gap_to_mlp_mean=('gap_to_mlp','mean'),
        gap_to_mlp_seed_sd=('gap_to_mlp','std')).reset_index().to_csv(out/'summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    cache=out/'.mpl-cache';cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache.resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    graph=table[(table.model=='sage') & (table.topology=='original')]
    for seed,part in graph.groupby('seed'):
        part=part.sort_values('two_hop_test_exposure')
        ax.plot(part.two_hop_test_exposure,part.gap_to_mlp,'o-' if part.two_hop_test_exposure.nunique()>1 else 'o',label=f'Seed {seed}')
    ax.axhline(0,color='gray',linewidth=1);ax.set_xlabel('Two-hop test exposure to training flows')
    ax.set_ylabel('GraphSAGE minus MLP macro-F1')
    ax.set_xlim(-.02,1.02)
    ax.set_title('Synthetic pipeline validation' if audit['synthetic_only'] else 'Connectivity sensitivity experiment')
    ax.legend();fig.tight_layout();fig.savefig(out/'exposure_gap.png',dpi=160);plt.close(fig)
    print(f'Run saved to {out}. '+('SYNTHETIC ONLY: no research conclusions.' if audit['synthetic_only'] else 'Review audit and controls before interpreting.'))

def main():
    parser=argparse.ArgumentParser(description='Exposure-aware graph IDS research prototype')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo');demo.add_argument('--out',default='data/synthetic');demo.add_argument('--rows',type=int,default=1200)
    demo.set_defaults(func=lambda a:print(__import__('exposure_ids.demo',fromlist=['make_demo']).make_demo(a.out,a.rows)))
    for name,func in [('audit',audit_command),('run',experiment)]:
        p=sub.add_parser(name);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
        p.add_argument('--protocol',choices=['random','inductive','chronological'],default='random')
        p.add_argument('--split-seed',type=int,default=42);p.add_argument('--graph-seed',type=int,default=77)
        p.add_argument('--window',type=float,default=60);p.add_argument('--neighbors',type=int,default=12)
        p.add_argument('--gap-seconds',type=float,default=0);p.add_argument('--per-file-limit',type=int)
        p.add_argument('--monthfirst',action='store_true');p.set_defaults(func=func)
        if name=='run':
            p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5]);p.add_argument('--levels',type=float,nargs='+',default=[0,.25,.5,.75,1])
            p.add_argument('--hidden',type=int,default=32);p.add_argument('--epochs',type=int,default=100)
            p.add_argument('--patience',type=int,default=15);p.add_argument('--rewire',action='store_true')
            p.add_argument('--degree-controlled',action='store_true')
    args=parser.parse_args()
    if getattr(args,'degree_controlled',False) and args.protocol!='random':
        parser.error('--degree-controlled is a random transductive diagnostic only')
    if getattr(args,'epochs',1)<1:parser.error('--epochs must be positive')
    if getattr(args,'gap_seconds',0)<0:parser.error('--gap-seconds must be nonnegative')
    if hasattr(args,'levels') and (any(v<0 or v>1 for v in args.levels) or len(set(args.levels))!=len(args.levels)):
        parser.error('--levels must be distinct fractions between zero and one')
    try: args.func(args)
    except (ValueError,FileNotFoundError,ImportError) as exc:parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
