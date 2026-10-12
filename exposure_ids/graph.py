from collections import defaultdict, deque
import numpy as np

def build_graph(frame, window_seconds=60, max_neighbors=12, causal=False):
    """Edges are source-message -> target; incoming candidate cap is fixed.

    Noncausal diagnostic graphs add reverse edges; total undirected degree is
    therefore not bounded by max_neighbors. No edges use labels or features.
    """
    if window_seconds <= 0 or max_neighbors < 1: raise ValueError('Invalid graph limits')
    times = frame.completion.astype('datetime64[ns]').astype('int64').to_numpy()
    order = np.argsort(times, kind='stable')
    recent = defaultdict(deque); edges = set()
    for i in order:
        candidates = set()
        hosts = {frame.src_ip.iloc[i], frame.dst_ip.iloc[i]}
        for host in hosts:
            queue = recent[host]
            while queue and times[i]-times[queue[0]] > window_seconds*1e9:
                queue.popleft()
            candidates.update(queue)
        for j in sorted(candidates, key=lambda j:(times[j],j), reverse=True)[:max_neighbors]:
            edges.add((int(j),int(i)))
            if not causal: edges.add((int(i),int(j)))
        for host in hosts:
            recent[host].append(int(i))
            # Only the most recent cap can qualify at a future step.
            while len(recent[host]) > max_neighbors: recent[host].popleft()
    return np.array(sorted(edges),dtype=np.int64).reshape(-1,2)

def partition_edges(edges, splits, policy, stage='test'):
    if not len(edges): return edges.copy()
    a,b = splits[edges[:,0]],splits[edges[:,1]]
    if policy == 'inductive': return edges[a == b]
    if policy == 'chronological': return edges.copy()
    # Validation and test are separate views: no hidden val -> test paths.
    allowed = {'train', 'val' if stage == 'val' else 'test'}
    return edges[np.isin(a,list(allowed)) & np.isin(b,list(allowed))]

def retain_cross_edges(edges, splits, fraction, seed):
    if not 0 <= fraction <= 1: raise ValueError('Retention fraction must be in [0,1]')
    if not len(edges): return edges.copy()
    cross = splits[edges[:,0]] != splits[edges[:,1]]
    # Paired arcs get the same decision; thresholds form nested graphs.
    pairs = sorted({tuple(sorted(e)) for e in edges[cross]})
    rng=np.random.default_rng(seed)
    retained={pair for pair,draw in zip(pairs,rng.random(len(pairs))) if draw < fraction}
    keep=[not c or tuple(sorted(e)) in retained for e,c in zip(edges,cross)]
    return edges[keep]

def exposure(frame, edges):
    splits = frame.split.to_numpy(); train = splits == 'train'; test = splits == 'test'
    train_hosts=set(frame.loc[train,'src_ip']) | set(frame.loc[train,'dst_ip'])
    test_hosts=set(frame.loc[test,'src_ip']) | set(frame.loc[test,'dst_ip'])
    reached=train.copy(); rates=[]
    for _ in range(2):
        nxt=reached.copy()
        if len(edges): nxt[edges[reached[edges[:,0]],1]]=True
        reached=nxt
        rates.append(float(reached[test].mean()) if test.any() else None)
    cross=int(np.sum(splits[edges[:,0]] != splits[edges[:,1]])) if len(edges) else 0
    indegree=np.bincount(edges[:,1],minlength=len(frame)) if len(edges) else np.zeros(len(frame),dtype=int)
    outdegree=np.bincount(edges[:,0],minlength=len(frame)) if len(edges) else indegree.copy()
    active=(train|test)|((indegree+outdegree)>0)
    return {'host_reuse':len(test_hosts & train_hosts)/len(test_hosts) if test_hosts else None,
            'cross_split_arc_fraction':cross/len(edges) if len(edges) else 0,
            'one_hop_test_exposure':rates[0],'two_hop_test_exposure':rates[1],
            'arcs':len(edges),'cross_split_arcs':cross,
            'view_nodes':int(active.sum()),'inactive_validation_nodes':int((~active).sum()),
            'mean_in_degree':float(indegree[active].mean()),'max_in_degree':int(indegree.max()),
            'mean_out_degree':float(outdegree[active].mean()),
            'isolated_nodes':int((((indegree+outdegree)==0)&active).sum())}

def rewire(edges, frame, window_seconds, causal, seed, swaps_per_arc=2):
    """Directed double-edge swaps preserve in/out degree and partition-pair counts.

    Undirected inputs are rewired as paired undirected edges. Temporal eligibility
    is enforced; shared-host semantics are deliberately disrupted. K-hop exposure
    is measured after rewiring and is NOT guaranteed invariant.
    """
    rng=np.random.default_rng(seed)
    base=edges if causal else np.array(sorted({tuple(sorted(e)) for e in edges}),dtype=np.int64).reshape(-1,2)
    items=[tuple(e) for e in base]; edge_set=set(items)
    times=frame.completion.astype('datetime64[ns]').astype('int64').to_numpy(); splits=frame.split.to_numpy()
    accepted=0; attempts=max(1,len(items)*swaps_per_arc*20)
    target=len(items)*swaps_per_arc
    for _ in range(attempts):
        if len(items)<2 or accepted>=target: break
        i,j=rng.choice(len(items),2,replace=False)
        a,b=items[i]; c,e=items[j]
        if len({a,b,c,e})<4: continue
        if causal:
            if splits[a]!=splits[c] or splits[b]!=splits[e]: continue
            new=[(a,e),(c,b)]
        else:
            # Orient endpoints so partition classes match.
            if splits[a]!=splits[c] or splits[b]!=splits[e]: continue
            new=[tuple(sorted((a,e))),tuple(sorted((c,b)))]
        if any(x in edge_set for x in new) or new[0]==new[1]: continue
        if any(abs(times[x]-times[y])>window_seconds*1e9 or (causal and times[x]>times[y]) for x,y in new): continue
        edge_set.remove(items[i]);edge_set.remove(items[j])
        items[i],items[j]=new;edge_set.update(new);accepted+=1
    result=sorted(edge_set) if causal else sorted(edge_set | {(b,a) for a,b in edge_set})
    return np.array(result,dtype=np.int64).reshape(-1,2), {'accepted_swaps':accepted,'target_swaps':target,
            'degree_preserved':True,'partition_pair_counts_preserved':True,'k_hop_exposure_preserved':False}

def reduce_cross_degree_preserving(edges, frame, fraction, window_seconds, seed):
    """Swap two train-test edges into within-split edges without changing degree.

    Eligibility may prevent reaching the target. New edges need not share hosts:
    this is a structural control, not an isolated intervention on familiarity.
    """
    if not 0 <= fraction <= 1: raise ValueError('Invalid degree-control target')
    splits=frame.split.to_numpy()
    items={tuple(sorted(e)) for e in edges}
    times=frame.completion.astype('datetime64[ns]').astype('int64').to_numpy()
    cross=[(a,b) if splits[a]=='train' else (b,a) for a,b in sorted(items) if splits[a]!=splits[b]]
    original=len(cross);target=int(round(original*fraction));rng=np.random.default_rng(seed)
    swaps=0;budget=max(1,original*50)
    for _ in range(budget):
        if len(cross)<=target or len(cross)<2:break
        i,j=rng.choice(len(cross),2,replace=False)
        a,b=cross[i];c,e=cross[j]
        if len({a,b,c,e})<4:continue
        new=[tuple(sorted((a,c))),tuple(sorted((b,e)))]
        if any(pair in items for pair in new):continue
        if any(abs(times[u]-times[v])>window_seconds*1e9 for u,v in new):continue
        items.remove(tuple(sorted((a,b))));items.remove(tuple(sorted((c,e))))
        items.update(new)
        for index in sorted([i,j],reverse=True):cross.pop(index)
        swaps+=1
    result=sorted(items | {(b,a) for a,b in items})
    return np.array(result,dtype=np.int64).reshape(-1,2),{
        'method':'degree_preserving_cross_reduction','target_cross_edges':target,
        'achieved_cross_edges':len(cross),'initial_cross_edges':original,
        'target_reached':len(cross)<=target,'accepted_swaps':swaps,
        'degree_preserved':True,'shared_host_semantics_preserved':False}
