import tempfile
import unittest
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from exposure_ids.demo import make_demo
from exposure_ids.data import load_flows, split_flows, preprocess
from exposure_ids.graph import build_graph, exposure, rewire, retain_cross_edges, reduce_cross_degree_preserving

class ResearchInvariants(unittest.TestCase):
    def fixture(self,n=120):
        f=pd.DataFrame({'src_ip':['a']*n,'dst_ip':['b']*n,
            'completion':pd.date_range('2017-07-04',periods=n,freq='s'),
            'split':np.where(np.arange(n)%2,'test','train')})
        return f

    def test_known_directed_exposure(self):
        f=self.fixture(4);f['split']=['train','test','test','test']
        e=np.array([[0,1],[1,2],[3,0]])
        stats=exposure(f,e)
        self.assertAlmostEqual(stats['one_hop_test_exposure'],1/3)
        self.assertAlmostEqual(stats['two_hop_test_exposure'],2/3)

    def test_causal_window_and_direction(self):
        f=self.fixture();e=build_graph(f,3,2,causal=True)
        times=f.completion.astype('datetime64[ns]').astype('int64').to_numpy()
        self.assertTrue(np.all(times[e[:,0]]<=times[e[:,1]]))
        self.assertTrue(np.all(times[e[:,1]]-times[e[:,0]]<=3e9))
        self.assertLessEqual(np.bincount(e[:,1]).max(),2)
        self.assertNotIn((0,10),set(map(tuple,e)))

    def test_nested_arc_pairs(self):
        f=self.fixture();e=build_graph(f)
        low=retain_cross_edges(e,f.split.to_numpy(),.25,3)
        high=retain_cross_edges(e,f.split.to_numpy(),.75,3)
        self.assertTrue(set(map(tuple,low))<=set(map(tuple,high)))
        self.assertEqual(set(map(tuple,low)),{(b,a) for a,b in low})

    def test_rewire_preserves_degrees(self):
        f=self.fixture();e=build_graph(f,60,3)
        r,info=rewire(e,f,60,False,9)
        self.assertGreater(info['accepted_swaps'],0)
        np.testing.assert_array_equal(np.bincount(e[:,1],minlength=len(f)),np.bincount(r[:,1],minlength=len(f)))
        self.assertEqual(len(e),len(r));self.assertFalse(any(a==b for a,b in r))

    def test_degree_control(self):
        f=self.fixture();e=build_graph(f,60,3)
        r,info=reduce_cross_degree_preserving(e,f,0,60,9)
        self.assertLess(info['achieved_cross_edges'],info['initial_cross_edges'])
        np.testing.assert_array_equal(np.bincount(e[:,1],minlength=len(f)),np.bincount(r[:,1],minlength=len(f)))

    def test_duplicates_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=make_demo(temp,300)
            f=pd.read_csv(path);f=pd.concat([f,f.iloc[[0]]],ignore_index=True);f.to_csv(path,index=False)
            data,features,audit=load_flows(temp)
            self.assertEqual(audit['exact_duplicate_rows'],1)
            data=split_flows(data,'random',42)
            self.assertEqual(data.groupby('group').split.nunique().max(),1)
            x,processor=preprocess(data,features)
            changed=data.copy();changed.loc[changed.split=='test',features[0]]=1e10
            _,processor2=preprocess(changed,features)
            self.assertEqual(processor,processor2)
            np.testing.assert_allclose(x[data.split=='train'].mean(axis=0),0,atol=1e-5)

    def test_metadata_missing_fails(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            pd.DataFrame({'Label':['BENIGN'],'Feature':[1]}).to_csv(Path(temp)/'bad.csv',index=False)
            with self.assertRaisesRegex(ValueError,'missing'):load_flows(temp)

    def test_duplicate_label_conflicts_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=make_demo(temp,300);f=pd.read_csv(path)
            conflict=f.iloc[[0]].copy();conflict['Label']='ATTACK'
            pd.concat([f,conflict],ignore_index=True).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'conflicting'):load_flows(temp)

    def test_chronological_ties_and_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            make_demo(temp,300);f,features,_=load_flows(temp)
            f=split_flows(f,'chronological',42,gap_seconds=2)
            tr=f.loc[f.split=='train','completion'];va=f.loc[f.split=='val','completion'];te=f.loc[f.split=='test','completion']
            self.assertLess(tr.max(),va.min());self.assertLess(va.max(),te.min())
            self.assertEqual(f.groupby('completion').split.nunique().max(),1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch installation required')
    def test_heldout_features_cannot_change_causal_training_logits(self):
        import torch
        from exposure_ids.models import MeanSAGE,adjacency
        f=self.fixture(20);f['split']=['train']*10+['test']*10
        e=build_graph(f,60,3,True);adj=adjacency(e,len(f))
        torch.manual_seed(1);model=MeanSAGE(3,4)
        x=torch.randn(20,3);changed=x.clone();changed[10:]=1e6
        torch.testing.assert_close(model(x,adj)[:10],model(changed,adj)[:10])

if __name__=='__main__':unittest.main()
