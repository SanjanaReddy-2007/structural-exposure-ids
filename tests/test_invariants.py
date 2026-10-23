import tempfile
import unittest
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from exposure_ids.demo import make_demo
from exposure_ids.data import load_flows, split_flows, preprocess
from exposure_ids.graph import build_graph, exposure, rewire, retain_cross_edges, reduce_cross_degree_preserving

def write_flows(folder, timestamps, durations=None, name='t.csv'):
    n = len(timestamps)
    pd.DataFrame({'Flow ID': [f'f{i}' for i in range(n)],
                  'Source IP': ['10.0.0.1'] * n, 'Destination IP': ['10.0.0.2'] * n,
                  'Timestamp': timestamps,
                  'Flow Duration': durations if durations is not None else [100] * n,
                  'Feature': list(range(n)),
                  'Label': ['BENIGN', 'ATTACK'] * (n // 2) + ['BENIGN'] * (n % 2)}
                 ).to_csv(Path(folder) / name, index=False)

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

    def test_blank_rows_dropped_and_counted_but_partial_rows_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=make_demo(temp,300);lines=Path(path).read_text().splitlines()
            width=lines[0].count(',')
            Path(path).write_text('\n'.join(lines[:50]+[','*width]*3+lines[50:])+'\n')
            data,features,audit=load_flows(temp)
            self.assertEqual(len(data),300)
            self.assertEqual(audit['fully_blank_rows_dropped'],{'synthetic.csv':3})
            self.assertIn('synthetic.csv:299',set(data.record_id))
            lines=Path(path).read_text().splitlines()
            Path(path).write_text('\n'.join(lines+[','.join(['1.1.1.1','2.2.2.2'])+','*(width-1)])+'\n')
            with self.assertRaisesRegex(ValueError,'invalid host, time or label'):load_flows(temp)

    def test_non_utf8_csv_is_decoded_and_encoding_recorded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=make_demo(temp,300);text=Path(path).read_text().replace('SYNTHETIC_ATTACK','Web Attack – Brute Force')
            Path(path).write_bytes(text.encode('cp1252'))
            data,features,audit=load_flows(temp)
            self.assertEqual(audit['file_encodings'],{'synthetic.csv':'cp1252'})
            self.assertIn('Web Attack – Brute Force',set(data.label))

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

    def test_twelve_hour_repair_is_applied_and_counted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            write_flows(temp,['6/7/2017 9:00','6/7/2017 12:05','6/7/2017 1:30','6/7/2017 5:00'])
            data,features,audit=load_flows(temp)
            expected=pd.to_datetime(['2017-07-06 09:00','2017-07-06 12:05','2017-07-06 13:30','2017-07-06 17:00'])
            self.assertEqual(data.timestamp.tolist(),list(expected))
            self.assertEqual(data.timestamp_original.tolist(),list(expected))
            info=audit['timestamp_files']['t.csv']
            self.assertEqual(info['pm_shifted_rows'],2)
            self.assertEqual(info['format'],'%d/%m/%Y %H:%M')
            self.assertEqual(info['hour_counts_before_repair'],{1:1,5:1,9:1,12:1})
            self.assertNotIn('timestamp_original',features)

    def test_24_hour_and_mixed_timestamp_formats_fail_loudly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            write_flows(temp,['6/7/2017 9:00','6/7/2017 17:00'])
            with self.assertRaisesRegex(ValueError,'hours outside 1-12'):load_flows(temp)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            write_flows(temp,['6/7/2017 9:00:15','6/7/2017 9:01'])
            with self.assertRaisesRegex(ValueError,'no single known format'):load_flows(temp)

    def test_floor_to_minute_keeps_original_and_counts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            write_flows(temp,['6/7/2017 9:00:30','6/7/2017 9:01:00'])
            data,_,audit=load_flows(temp,floor_to_minute=True)
            self.assertEqual(data.timestamp.tolist(),list(pd.to_datetime(['2017-07-06 09:00:00','2017-07-06 09:01:00'])))
            self.assertEqual(data.timestamp_original.iloc[0],pd.Timestamp('2017-07-06 09:00:30'))
            self.assertEqual(audit['timestamp_files']['t.csv']['floored_rows'],1)
            self.assertTrue(audit['floor_to_minute'])

    def test_negative_duration_strict_by_default_and_recorded_when_dropped(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            write_flows(temp,['6/7/2017 9:00','6/7/2017 9:01','6/7/2017 9:02','6/7/2017 9:03'],durations=[100,-5,0,7])
            with self.assertRaisesRegex(ValueError,'Invalid Flow Duration'):load_flows(temp)
            data,_,audit=load_flows(temp,drop_invalid_duration=True)
            self.assertEqual(len(data),3)
            fd=audit['flow_duration']
            self.assertEqual(fd['negative_rows'],1);self.assertEqual(fd['dropped_rows'],1)
            self.assertEqual(fd['dropped_record_ids'],['t.csv:1'])
            self.assertEqual(fd['dropped_per_file'],{'t.csv':1})
            self.assertEqual(fd['rows_before_filter'],4)
            self.assertIn(0,pd.to_numeric(data.flow_duration).tolist())  # zero duration is kept

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