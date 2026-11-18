"""Canonical runner integration and method UV installs, entirely in temp roots.

The fake uv is a real child process: these tests exercise argv/environment,
registry migration, cancellation and cleanup without installing packages.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from unittest import mock

import numpy as np

from tsad_harness import environments, runner
from tsad_harness.config import HarnessError

REPO = Path(__file__).resolve().parents[3]


class EvaluationResourceTests(unittest.TestCase):
    @contextmanager
    def gpu_snapshot(self, *, mask='0,1,2', compute_apps='GPU-busy\n', rows=None):
        rows = rows if rows is not None else (
            '0, GPU-small, 8192, 128, 8064, 0\n'
            '1, GPU-large, 24576, 128, 24448, 0\n'
            '2, GPU-busy, 49152, 64, 49088, 0\n')
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':mask}), \
             mock.patch.object(runner.subprocess, 'run', side_effect=[
                 SimpleNamespace(stdout=rows), SimpleNamespace(stdout=compute_apps)]) as query:
            yield query

    def test_cpu_method_does_not_reserve_requested_gpu(self):
        with mock.patch.object(runner, 'discover_resources', return_value={'gpus': ''}) as discover:
            result = runner.evaluation_resources({'compute_device': 'cpu'}, '1')
        discover.assert_called_once_with('cpu')
        self.assertEqual(result['gpus'], '')

    def test_cuda_method_waits_instead_of_launching_without_cuda(self):
        with mock.patch.object(runner, 'discover_resources', return_value={'gpus': ''}):
            with self.assertRaisesRegex(HarnessError, 'requires CUDA'):
                runner.evaluation_resources({'compute_device': 'cuda'})

    def test_legacy_method_keeps_automatic_selection(self):
        with mock.patch.object(runner, 'discover_resources', return_value={'gpus': '1'}) as discover:
            result = runner.evaluation_resources({}, '1')
        discover.assert_called_once_with('1')
        self.assertEqual(result['gpus'], '1')

    def test_automatic_selection_ranks_idle_gpus_by_free_memory(self):
        with self.gpu_snapshot():
            result = runner.discover_resources()
        self.assertEqual(result['gpus'], '1,0')
        self.assertTrue(result['devices'][2]['compute_active'])
        self.assertFalse(result['devices'][2]['idle'])
        self.assertEqual(result['idle_policy'],
                         {'max_used_memory_mib':256, 'max_utilization_gpu_percent':5})

    def test_evaluation_minimum_filters_idle_gpus_and_accepts_exact_capacity(self):
        for minimum, expected in ((8064, '1,0'), (20000, '1')):
            with self.subTest(minimum=minimum), self.gpu_snapshot():
                result = runner.evaluation_resources({'min_gpu_memory_mib':minimum})
            self.assertEqual(result['gpus'], expected)
            self.assertEqual(result['min_gpu_memory_mib'], minimum)

    def test_insufficient_idle_memory_is_retryable(self):
        with self.gpu_snapshot():
            with self.assertRaisesRegex(HarnessError, r'RESOURCE_EXECUTION.*min_gpu_memory_mib=30000'):
                runner.evaluation_resources({'min_gpu_memory_mib':30000})

    def test_preferred_gpus_only_reorder_eligible_idle_devices(self):
        for preference in ('0', 'GPU-small', '2,0', '99,GPU-small'):
            with self.subTest(preference=preference), self.gpu_snapshot():
                result = runner.evaluation_resources({'preferred_gpus':preference})
            self.assertEqual(result['gpus'], '0,1')
            self.assertEqual(result['preferred_gpus'], preference)

    def test_preference_cannot_override_minimum_or_visibility(self):
        with self.gpu_snapshot():
            result = runner.evaluation_resources({'preferred_gpus':'0', 'min_gpu_memory_mib':20000})
        self.assertEqual(result['gpus'], '1')
        with self.gpu_snapshot(mask='0'):
            result = runner.evaluation_resources({'preferred_gpus':'1'})
        self.assertEqual(result['gpus'], '0')

    def test_explicit_lease_is_rechecked_without_switching_to_preferred_gpu(self):
        evaluation = {'compute_device':'cuda', 'preferred_gpus':'1', 'min_gpu_memory_mib':8000}
        with self.gpu_snapshot(mask='1'):
            result = runner.evaluation_resources(evaluation, '0')
        self.assertEqual(result['gpus'], '0')
        with self.gpu_snapshot():
            with self.assertRaisesRegex(HarnessError, 'min_gpu_memory_mib'):
                runner.evaluation_resources(dict(evaluation, min_gpu_memory_mib=20000), '0')
        with self.gpu_snapshot(compute_apps='GPU-small\n'):
            with self.assertRaisesRegex(HarnessError, 'busy'):
                runner.evaluation_resources(evaluation, '0')

    def test_explicit_gpu_order_and_all_requested_memory_requirements_are_preserved(self):
        with self.gpu_snapshot():
            result = runner.discover_resources('0,1', preferred_gpus='1')
        self.assertEqual(result['gpus'], '0,1')
        with self.gpu_snapshot():
            with self.assertRaisesRegex(HarnessError, 'min_gpu_memory_mib'):
                runner.discover_resources('0,1', min_gpu_memory_mib=20000)

    def test_execution_settings_do_not_relax_idle_measurements(self):
        for used, free, utilization in ((257, 24000, 0), (0, 24000, 6), (0, '[N/A]', 0)):
            rows = f'0, GPU-small, 24576, {used}, {free}, {utilization}\n'
            with self.subTest(rows=rows), self.gpu_snapshot(mask='0', rows=rows, compute_apps=''):
                with self.assertRaisesRegex(HarnessError, 'idle state cannot be verified'):
                    runner.evaluation_resources({'preferred_gpus':'0', 'min_gpu_memory_mib':1000})

    def test_compute_process_query_failure_remains_retryable(self):
        with self.gpu_snapshot() as query:
            query.side_effect = [SimpleNamespace(stdout='0, GPU-small, 8192, 0, 8192, 0\n'),
                                 subprocess.CalledProcessError(1, 'nvidia-smi')]
            with self.assertRaisesRegex(HarnessError, 'cannot verify GPU compute processes'):
                runner.evaluation_resources({'preferred_gpus':'0'})

    def test_minimum_prevents_cpu_fallback_when_cuda_is_unavailable(self):
        with mock.patch.object(runner.subprocess, 'run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(HarnessError, 'min_gpu_memory_mib'):
                runner.evaluation_resources({'min_gpu_memory_mib':1000})

    def test_cpu_evaluation_ignores_gpu_execution_settings(self):
        with self.gpu_snapshot() as query:
            result = runner.evaluation_resources({'compute_device':'cpu', 'min_gpu_memory_mib':30000,
                                                  'preferred_gpus':'1'}, '1')
        query.assert_not_called()
        self.assertEqual(result['gpus'], '')

    def test_invalid_execution_settings_are_rejected_before_discovery(self):
        invalid = [{'min_gpu_memory_mib':value} for value in
                   (-1, True, '1000', None, float('inf'), float('nan'))]
        invalid.extend({'preferred_gpus':value} for value in ([0], 1, 'cpu'))
        for evaluation in invalid:
            with self.subTest(evaluation=evaluation), mock.patch.object(runner.subprocess, 'run') as query:
                with self.assertRaises(HarnessError):
                    runner.evaluation_resources(evaluation)
            query.assert_not_called()


class CanonicalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.methods = self.root/'EasyTSAD/Methods'
        class FixtureConfig:
            def __getitem__(self, key):
                return {'schema':'naive','datasets':['Tiny']}
        self.cfg = FixtureConfig()
        self.cfg.methods_dir = self.methods
        self.patch = mock.patch.multiple(runner,ROOT=self.root,RUNS=self.root/'LiveLeaderBoard/results/runs',load=lambda:self.cfg)
        self.patch.start()
    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()
    def manifest(self, attempt='current', status='success', value=.7):
        directory = self.methods/'Example/reproduction/runs'/attempt/'naive/Tiny'
        directory.mkdir(parents=True,exist_ok=True)
        scores = self.root/'Results/Scores/Example/naive/Tiny'
        scores.mkdir(parents=True,exist_ok=True)
        np.save(scores/'selected.npy',[.1,.9])
        np.save(scores/'previous.npy',[.2,.8])
        native = self.root/'Results/Evals/Example/naive/Tiny'
        native.mkdir(parents=True,exist_ok=True)
        (native/'avg.json').write_text(json.dumps({'best f1 under pa':{'f1':value}}))
        (native/'all.json').write_text(json.dumps({'selected':{'best f1 under pa':{'f1':value}}}))
        record = dict(result_format='easytsad-v1',method='Example',schema='naive',dataset='Tiny',
            attempt_id=attempt,run_id='20260101T000000Z',status=status,started_at='2026-01-01T00:00:00Z',
            finished_at='2026-01-01T00:00:10Z' if status=='success' else None,
            score_dir=str(scores.relative_to(self.root)),all_path=str((native/'all.json').relative_to(self.root)),
            avg_path=str((native/'avg.json').relative_to(self.root)),runtime_path='Results/RunTime/Example/naive/Tiny/time.json',
            scored_curves=['selected'],n_curves_scored=1,n_curves_evaluated=1,total_curves=1,
            metric_keys={'point_f1':'best f1 under pa'},partial=True,canonical_metrics_committed=True,
            diagnostics={'constant':False,'all_zero':False,'core_f1_zero':False},config={'curves':['selected']})
        (directory/'manifest.json').write_text(json.dumps(record))
        return directory,record
    def test_current_attempt_lookup_excludes_old_legacy_and_wrong_identity(self):
        current,_ = self.manifest(status='running')
        old = runner.RUNS/'Example/naive/Tiny/20260101T000000Z'
        old.mkdir(parents=True)
        (old/'manifest.json').write_text(json.dumps(dict(method='Example',schema='naive',dataset='Tiny',
            run_id='20260101T000000Z',status='success',finished_at='2099-01-01T00:00:00Z')))
        self.assertEqual(runner.find_run('Example','naive','Tiny',attempt_id='current'),current)
        self.assertIsNone(runner.find_run('Example','naive','Tiny',attempt_id='missing'))
        self.assertIsNone(runner.find_run('Example','naive','Tiny',attempt_id='current',run_id='other'))
    def test_score_check_filters_fresh_subset_and_rejects_missing_or_nonfinite(self):
        directory,manifest = self.manifest()
        self.assertTrue(runner.check_scores(directory,expected=1)['ok'])
        scores = self.root/manifest['score_dir']
        np.save(scores/'selected.npy',[np.nan,np.inf])
        self.assertFalse(runner.check_scores(directory,expected=1)['ok'])
        (scores/'selected.npy').unlink()
        check = runner.check_scores(directory,expected=1)
        self.assertFalse(check['ok'])
        self.assertTrue(any('missing' in problem for problem in check['problems']))
    def test_partial_attempt_progress_wins_over_older_complete_average(self):
        directory,manifest = self.manifest(value=.9)
        manifest['canonical_metrics_committed'] = False
        (directory/'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(runner.read_summary(directory),{})
        rows = [{'type':'curve_evaluated','data':{'summary':{'point_f1':{'f1':.3}}}}]
        (directory/'progress.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n{"data":')
        self.assertEqual(runner.read_summary(directory),{'point_f1':{'f1':.3}})
    def test_eval_status_has_native_refs_counts_and_diagnostics(self):
        directory,manifest = self.manifest()
        task_dir = self.root/'tasks/example'
        task_dir.mkdir(parents=True)
        (task_dir/'eval.out').write_text('run_id=20260101T000000Z\n')
        task = SimpleNamespace(method='Example',dir=task_dir,state={'eval':{'pid':123,'log':'eval.out','attempt_id':'current','datasets':['Tiny']}})
        with mock.patch.object(runner,'eval_alive',return_value=False):
            status = runner.eval_status(task)
        self.assertTrue(status['done'])
        data = status['datasets']['Tiny']
        self.assertEqual(data['diagnostics'],manifest['diagnostics'])
        self.assertEqual(data['avg_path'],manifest['avg_path'])
        self.assertEqual(data['metrics_summary'],{'point_f1':{'f1':.7}})
        self.assertEqual(data['n_curves_scored'],1)
        self.assertTrue(data['manifest'].endswith('/manifest.json'))
    def test_source_hash_prunes_local_venv_and_keeps_upstream_sources(self):
        method = self.methods/'Example'
        method.mkdir(parents=True)
        (method/'Example.py').write_text('model=1')
        (method/'upstream').mkdir()
        (method/'upstream/model.py').write_text('official=1')
        original = runner.source_hash(method)
        for subdir in ('environment/.venv/lib','reproduction/runs/attempt','upstream/.git'):
            path = method/subdir
            path.mkdir(parents=True)
            (path/'generated.py').write_text('generated=1')
        self.assertEqual(runner.source_hash(method),original)
        (method/'upstream/model.py').write_text('official=2')
        self.assertNotEqual(runner.source_hash(method),original)

    def test_generated_eval_keeps_auxiliary_outputs_in_attempt_workdir(self):
        task = SimpleNamespace(method='Example',dir=self.root/'tasks/example')
        config = {'eval':{'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                          'evals':['point_f1'],'seed':2026}}
        with mock.patch.object(runner,'load',return_value=config):
            path = runner.write_spec(task,'eval')
        spec = tomllib.loads(path.read_text())
        self.assertTrue(spec['isolate_workdir'])
        self.assertEqual(spec['datasets'],['Tiny'])

    def test_gpu_execution_feedback_preserves_identity_and_official_parameters(self):
        method = self.methods/'Example'
        method.mkdir(parents=True)
        config_path = method/'config.toml'
        official = '[Model_Params.Default]\nepochs=100\nwindow=64\n'
        config_path.write_text(official)
        task = SimpleNamespace(method='Example',method_dir=method,dir=self.root/'tasks/example')
        plan = {'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                'evals':['point_f1'],'seed':2026}
        adaptive = dict(plan, min_gpu_memory_mib=20000, preferred_gpus='1')
        self.assertEqual(runner.evaluation_identity(task, plan, 'Tiny'),
                         runner.evaluation_identity(task, adaptive, 'Tiny'))
        changed = dict(plan, overrides={'Example':{'hparams':{'epochs':50}}})
        self.assertNotEqual(runner.evaluation_identity(task, plan, 'Tiny'),
                            runner.evaluation_identity(task, changed, 'Tiny'))
        with mock.patch.object(runner,'load',return_value={'eval':plan}):
            spec = tomllib.loads(runner.write_spec(task,'eval',evaluation=adaptive).read_text())
        self.assertNotIn('overrides', spec)
        self.assertNotIn('min_gpu_memory_mib', spec)
        self.assertNotIn('preferred_gpus', spec)
        self.assertEqual(config_path.read_text(), official)

    def test_launch_records_exact_detached_process_and_task_relative_paths(self):
        task_dir = self.root/'tasks/example'
        (task_dir/'runs').mkdir(parents=True)
        spec = task_dir/'runs/eval.toml'
        spec.write_text('schema="naive"')
        script = self.root/'fake-tsad'
        script.write_text('#!/bin/sh\nprintf "run_id=20260101T000000Z\\n"\n')
        script.chmod(0o755)
        task = SimpleNamespace(method='Example',dir=task_dir,state={})
        task.set = lambda **fields: task.state.update(fields)
        task.log = lambda *args,**kwargs:None
        self.cfg.network_env = lambda:dict(os.environ)
        actual_popen = subprocess.Popen
        processes = []
        def spawn(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            processes.append(process)
            return process
        with mock.patch.object(runner,'write_spec',return_value=spec),mock.patch.object(runner,'TSAD',script), \
             mock.patch.object(runner,'discover_resources',return_value={'gpus':'','devices':[]}), \
             mock.patch.object(runner.subprocess,'Popen',side_effect=spawn):
            info = runner.launch_eval(task,gpus='cpu',datasets=['Tiny'],attempt_id='current')
        processes[0].wait(timeout=5)
        self.assertIsNotNone(info['processIdentity'])
        self.assertTrue(info['commandVerified'])
        self.assertEqual(info['argv'],['nohup',str(script),'exp',str(spec),'--methods','Example','--force'])
        self.assertEqual(info['cwd'],'.')
        self.assertEqual(info['attempt_id'],'current')
        self.assertEqual(info['datasets'],['Tiny'])
        self.assertEqual(info['spec'],'runs/eval.toml')
        self.assertFalse(Path(info['log']).is_absolute())
        self.assertIn('run_id=20260101T000000Z',(task_dir/info['log']).read_text())
        self.assertEqual(task.state['eval'],info)

    def test_launch_rechecks_leased_gpu_memory_before_writing_spec_or_spawning(self):
        task = SimpleNamespace(method='Example',dir=self.root/'tasks/example',state={})
        plan = {'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                'evals':['point_f1'],'compute_device':'cuda',
                'min_gpu_memory_mib':20000,'preferred_gpus':'1'}
        rows = '0, GPU-small, 8192, 128, 8064, 0\n1, GPU-large, 24576, 128, 24448, 0\n'
        with mock.patch.object(runner,'load',return_value={'eval':plan}), \
             mock.patch.object(runner.subprocess,'run',side_effect=[
                 SimpleNamespace(stdout=rows), SimpleNamespace(stdout='')]), \
             mock.patch.object(runner,'write_spec') as write, \
             mock.patch.object(runner.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(HarnessError, 'min_gpu_memory_mib=20000'):
                runner.launch_eval(task,gpus='0')
        write.assert_not_called()
        spawn.assert_not_called()
        self.assertEqual(task.state, {})

    def test_task_evaluation_config_and_spec_do_not_mutate_inputs(self):
        defaults = {'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                    'evals':['point_f1'],'seed':2026}
        overrides = {'schema':'all_in_one','dataset_type':'MTS','datasets':['TaskData'],
                     'evals':['point_auprc'],'seed':17}
        task = SimpleNamespace(method='Example',dir=self.root/'tasks/example',
                               spec={'evaluation':overrides})
        with mock.patch.object(runner,'load',return_value={'eval':defaults}):
            effective = runner.evaluation_config(task)
            self.assertEqual(effective,overrides)
            effective['datasets'].append('Detached')
            effective['evals'].append('Detached')
            spec = tomllib.loads(runner.write_spec(task,'eval').read_text())
        for key in overrides:
            self.assertEqual(spec[key],overrides[key])
        self.assertEqual(defaults['datasets'],['Tiny'])
        self.assertEqual(overrides['datasets'],['TaskData'])
        self.assertEqual(overrides['evals'],['point_auprc'])

    def test_persisted_task_evaluation_overrides_spec_and_preserves_seed(self):
        defaults = {'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                    'evals':['point_f1'],'seed':2026}
        persisted = {'schema':'all_in_one','dataset_type':'MTS','datasets':['Persisted'],
                     'evals':['point_auprc'],'seed':42}
        task = SimpleNamespace(method='Example',dir=self.root/'tasks/example',
                               spec={'evaluation':{'seed':17,'datasets':['Spec']}},
                               state={'evaluation':persisted})
        with mock.patch.object(runner,'load',return_value={'eval':defaults}):
            effective = runner.evaluation_config(task)
            spec = tomllib.loads(runner.write_spec(task,'eval').read_text())
        self.assertEqual(effective,persisted)
        for key in persisted:
            self.assertEqual(spec[key],persisted[key])
        effective['datasets'].append('Detached')
        self.assertEqual(persisted['datasets'],['Persisted'])

    def test_launch_persists_effective_evaluation_with_explicit_dataset_precedence(self):
        task = SimpleNamespace(method='Example',dir=self.root/'tasks/example',state={},
                               spec={'evaluation':{'schema':'all_in_one','datasets':['TaskData']}})
        task.set = lambda **fields: task.state.update(fields)
        task.log = mock.Mock()
        defaults = {'schema':'naive','dataset_type':'UTS','datasets':['Tiny'],
                    'evals':['point_f1'],'seed':2026}
        class LaunchConfig:
            def __getitem__(self, key):
                return defaults
            def network_env(self):
                return {}
        config = LaunchConfig()
        evaluation = {'dataset_type':'MTS','datasets':['LaunchData'],'evals':['point_auprc'],'seed':42}
        with mock.patch.object(runner,'load',return_value=config), \
             mock.patch.object(runner,'discover_resources',return_value={'gpus':''}), \
             mock.patch.object(runner.subprocess,'Popen',return_value=SimpleNamespace(pid=123)) as spawn, \
             mock.patch.dict(sys.modules,{'tsad_harness.scheduler':
                                          SimpleNamespace(process_identity=lambda pid:'fixture')}):
            info = runner.launch_eval(task,gpus='cpu',datasets=['Explicit'],evaluation=evaluation)
        spec = tomllib.loads((task.dir/info['spec']).read_text())
        expected = dict(defaults,schema='all_in_one',**evaluation)
        expected['datasets'] = ['Explicit']
        self.assertEqual(info['evaluation'],expected)
        self.assertEqual(info['datasets'],['Explicit'])
        for key in expected:
            self.assertEqual(spec[key],expected[key])
        self.assertEqual(evaluation['datasets'],['LaunchData'])
        spawn.assert_called_once()

    def test_eval_status_uses_persisted_evaluation_after_defaults_change(self):
        task_dir = self.root/'tasks/example'
        task_dir.mkdir(parents=True)
        (task_dir/'eval.out').write_text('run_id=20260101T000000Z\n')
        task = SimpleNamespace(method='Example',dir=task_dir,state={'eval':{
            'pid':123,'log':'eval.out','datasets':['Old'],
            'evaluation':{'schema':'all_in_one','dataset_type':'MTS',
                          'evals':['point_auprc'],'datasets':['Persisted']}}})
        with mock.patch.object(runner,'eval_alive',return_value=False), \
             mock.patch.object(runner,'find_run',return_value=None) as find:
            status = runner.eval_status(task)
        find.assert_called_once_with('Example','all_in_one','Persisted',
                                     run_id='20260101T000000Z',attempt_id=None)
        self.assertEqual(list(status['datasets']),['Persisted'])
        self.assertEqual(status['schema'],'all_in_one')
        self.assertEqual(status['dataset_type'],'MTS')

    def test_pid_reuse_does_not_attach_to_an_unrelated_process(self):
        task = SimpleNamespace(state={'eval':{'pid':123,'processIdentity':'expected'}})
        with mock.patch('tsad_harness.scheduler.process_identity',return_value='different'),mock.patch.object(runner,'_alive') as alive:
            self.assertFalse(runner.eval_alive(task))
            alive.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('statsmodels'), 'requires runner evaluation dependencies (statsmodels)')
class PublicationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(sys, 'path', [str(REPO/'LiveLeaderBoard/pipeline'), *sys.path]):
            module_spec = importlib.util.spec_from_file_location(
                'fixture_publication_run', REPO/'LiveLeaderBoard/pipeline/run.py')
            cls.pipeline = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(cls.pipeline)
            import export
            cls.exporter = export

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.methods = self.root/'EasyTSAD/Methods'
        (self.methods/'Example').mkdir(parents=True)
        self.run_dir = self.methods/'Example/reproduction/runs/current/naive/Tiny'
        self.scores = self.root/'Results/Scores/Example/naive/Tiny'
        self.evals = self.root/'Results/Evals/Example/naive/Tiny'
        self.spec = {'schema':'naive', 'datasets':['Tiny'], 'evals':['point_f1','event_f1','auprc'],
                     'isolate_workdir':True, 'global_cfg':{'EvalSetting':{'use_margin':False}}}
        self.labels = {}
        for curve in ('first', 'second', 'normal'):
            directory = self.root/'datasets/UTS/Tiny'/curve
            directory.mkdir(parents=True)
            labels = np.zeros(12)
            if curve != 'normal':
                labels[4:7] = 1
            self.labels[curve] = labels
            for name, values in {'train':np.arange(12.), 'test':np.arange(12.),
                                 'train_label':np.zeros(12), 'test_label':labels}.items():
                np.save(directory/(name+'.npy'), values)
            (directory/'info.json').write_text('{}')
        self.model = SimpleNamespace(param_statistic=mock.Mock(), train_valid_phase=mock.Mock(),
                                     test_phase=mock.Mock(), anomaly_score=mock.Mock(
                                         return_value=np.array([0.,1.,0.,1.,4.,5.,4.,1.,0.,1.,0.,1.])))
        from EasyTSAD.Methods import BaseMethodMeta
        patches = [
            mock.patch.multiple(self.pipeline, ROOT=self.root, METHODS_DIR=self.methods,
                                RESULTS_DIR=self.root/'LiveLeaderBoard/results',
                                RUNS_DIR=self.root/'LiveLeaderBoard/results/runs',
                                git_info=lambda:{}, env_info=lambda:{}, set_seed=lambda seed:None),
            mock.patch.multiple(self.exporter, ROOT=self.root,
                                RESULTS_DIR=self.root/'LiveLeaderBoard/results',
                                RUNS_DIR=self.root/'LiveLeaderBoard/results/runs',
                                CATALOG=self.root/'LiveLeaderBoard/catalog.toml'),
            mock.patch.object(self.pipeline.PathManager, 'get_default_config_path',
                              side_effect=lambda method:str(self.methods/method/'config.toml')),
            mock.patch.dict(BaseMethodMeta.registry, {'Example':lambda hparams:self.model}),
            mock.patch.dict(sys.modules, {'EasyTSAD.Methods.Example':SimpleNamespace(),
                                          'EasyTSAD.Methods.Example.Example':SimpleNamespace()}),
            mock.patch.dict(os.environ, {'TSAD_ATTEMPT_ID':'current', 'TSAD_ENV':''}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        previous_pm = self.pipeline.PathManager.get_instance()
        self.addCleanup(setattr, self.pipeline.PathManager, '_instance', previous_pm)

    def manifest(self):
        return json.loads((self.run_dir/'manifest.json').read_text())

    def fail_publication(self):
        @contextmanager
        def fail_lock():
            persisted = self.manifest()
            self.assertEqual(persisted['execution_phase'], 'publication')
            self.assertEqual(persisted['scored_curves'], ['first','normal','second'])
            self.assertEqual(persisted['n_curves_scored'], 3)
            self.assertEqual(persisted['n_curves_evaluated'], 2)
            raise OSError(22, 'Invalid argument')
            yield
        with mock.patch.object(self.pipeline, 'export_lock', fail_lock):
            self.assertEqual(self.pipeline.run_one(self.spec, 'Example', 'Tiny', 'original', True), 'failed')
        self.model.train_valid_phase.reset_mock()
        self.model.test_phase.reset_mock()
        self.model.anomaly_score.reset_mock()
        return self.manifest()

    def recover(self, run_id='recovered', force=True, **spec):
        with mock.patch.object(self.pipeline.Naive, 'do_exp',
                               side_effect=AssertionError('recovery must not train')) as train, \
             mock.patch.object(self.pipeline.Naive, 'load_data',
                               side_effect=AssertionError('recovery must not load training data')):
            result = self.pipeline.run_one(dict(self.spec, recover_publication=True, **spec),
                                           'Example', 'Tiny', run_id, force)
        train.assert_not_called()
        self.model.train_valid_phase.assert_not_called()
        self.model.test_phase.assert_not_called()
        self.model.anomaly_score.assert_not_called()
        return result

    def test_publication_failure_replays_scores_with_new_run_id_and_native_metrics(self):
        previous = self.fail_publication()
        score_bytes = {path.name:path.read_bytes() for path in self.scores.glob('*.npy')}
        runtime = self.root/previous['runtime_path']
        original_runtime = runtime.read_bytes()
        with mock.patch.dict(self.pipeline.SCHEMAS, {'naive':mock.Mock(
                side_effect=AssertionError('recovery must not construct a training schema'))}):
            self.assertEqual(self.recover(require_cuda=True, monitor_gpu=True), 'success')
        recovered = self.manifest()
        self.assertEqual(recovered['run_id'], 'recovered')
        self.assertEqual(recovered['recovery_source_run_id'], 'original')
        backup = json.loads((self.run_dir/recovered['recovery_source_manifest']).read_text())
        self.assertEqual(backup, previous)
        self.assertEqual(recovered['config_hash'], previous['config_hash'])
        self.assertEqual(recovered['config'], previous['config'])
        self.assertEqual(recovered['n_curves_scored'], 3)
        self.assertEqual(recovered['n_curves_evaluated'], 2)
        self.assertEqual(recovered['skipped_all_normal'], ['normal'])
        self.assertTrue(recovered['canonical_metrics_committed'])
        self.assertEqual({path.name:path.read_bytes() for path in self.scores.glob('*.npy')}, score_bytes)
        self.assertEqual(runtime.read_bytes(), original_runtime)
        native = json.loads((self.evals/'all.json').read_text())
        average = json.loads((self.evals/'avg.json').read_text())
        events = [json.loads(line) for line in (self.run_dir/'progress.jsonl').read_text().splitlines()]
        original_summary = [row['data']['summary'] for row in events
                            if row['type'] == 'dataset_finished' and row['run_id'] == 'original'][0]
        for key, name in recovered['metric_keys'].items():
            expected_average = original_summary[key]
            self.assertEqual(average[name], expected_average if 'value' not in expected_average else expected_average['value'])
        expected = {}
        for curve in ('first','second'):
            performance = self.pipeline.Performance('Example', 'Tiny', curve,
                np.load(self.scores/(curve+'.npy')), self.labels[curve].copy(), (0,0))
            _, expected[curve] = performance.perform_eval([self.pipeline.EVALS[key]() for key in self.spec['evals']])
        self.assertEqual(native, expected)
        self.assertEqual(self.pipeline.existing_success('Example','naive','Tiny', recovered['config_hash']), self.run_dir)
        self.assertEqual(self.pipeline.run_one(self.spec, 'Example', 'Tiny', 'later', False), 'skipped')
        self.assertEqual(self.manifest()['run_id'], 'recovered')

    def test_legacy_export_lock_failure_uses_only_original_run_progress(self):
        previous = self.fail_publication()
        previous.pop('execution_phase')
        previous.pop('scored_curves')
        previous['error'] = ('OSError: [Errno 22] Invalid argument\n'
                             '  File "run.py", line 461, in run_one\n    with export_lock():\n'
                             '  File "export.py", line 155, in export_lock\n'
                             '    fcntl.flock(lock, fcntl.LOCK_EX)\nOSError: [Errno 22] Invalid argument\n')
        self.pipeline.write_json(self.run_dir/'manifest.json', previous)
        with (self.run_dir/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'type':'curve_evaluated', 'run_id':'unrelated',
                                    'method':'Example','schema':'naive','dataset':'Tiny','curve':'stale'})+'\n')
            stream.write('{"interrupted":')
        with mock.patch.object(self.pipeline, 'existing_success',
                               side_effect=AssertionError('explicit recovery must not skip')):
            self.assertEqual(self.recover(force=False), 'success')
        self.assertEqual(self.manifest()['scored_curves'], ['first','normal','second'])

    def test_recovery_refuses_insufficient_evidence_and_never_retrains(self):
        previous = self.fail_publication()
        cases = [({'config_hash':'different'}, 'config_hash mismatch'),
                 ({'status':'success'}, 'failed manifest'),
                 ({'execution_phase':'execution'}, 'explicit publication failure'),
                 ({'execution_phase':None, 'error':'training failed with export_lock text'}, 'explicit publication failure'),
                 ({'n_curves_scored':2}, 'complete scored curve coverage'),
                 ({'attempt_id':'other'}, 'same attempt'),
                 ({'scored_curves':['first','first','normal']}, 'unique scored curve list'),
                 ({'scored_curves':['first','normal','stale']}, 'missing score files')]
        for changes, message in cases:
            with self.subTest(changes=changes):
                self.pipeline.write_json(self.run_dir/'manifest.json', dict(previous, **changes))
                self.assertEqual(self.recover(), 'failed')
                self.assertIn(message, self.manifest()['error'])
                self.assertFalse((self.evals/'avg.json').exists())
        self.pipeline.write_json(self.run_dir/'manifest.json', previous)
        (self.scores/'first.npy').unlink()
        self.assertEqual(self.recover(), 'failed')
        self.assertIn('missing score files', self.manifest()['error'])

    def test_recovery_refuses_invalid_arrays_and_changed_dataset_coverage(self):
        previous = self.fail_publication()
        for values in ([np.nan], [], [[1.]], list(range(13)), ['invalid']):
            with self.subTest(values=values):
                self.pipeline.write_json(self.run_dir/'manifest.json', previous)
                np.save(self.scores/'first.npy', values)
                self.assertEqual(self.recover(), 'failed')
                self.assertFalse((self.evals/'avg.json').exists())
        self.pipeline.write_json(self.run_dir/'manifest.json', previous)
        (self.root/'datasets/UTS/Tiny/normal').rename(self.root/'datasets/UTS/Tiny/changed')
        self.assertEqual(self.recover(), 'failed')
        self.assertIn('expected dataset coverage', self.manifest()['error'])

    def test_repeated_publication_failure_remains_recoverable(self):
        self.fail_publication()
        with mock.patch.object(self.pipeline, 'export_lock', side_effect=OSError(22, 'Invalid argument')):
            self.assertEqual(self.recover(), 'failed')
        failed = self.manifest()
        self.assertEqual(failed['execution_phase'], 'publication')
        self.assertEqual(failed['scored_curves'], ['first','normal','second'])
        self.assertEqual(self.recover(run_id='recovered-again'), 'success')
        self.assertEqual(self.manifest()['recovery_source_run_id'], 'recovered')

    def test_recovery_without_manifest_fails_instead_of_training(self):
        self.assertEqual(self.recover(), 'failed')
        self.assertIn('cannot read or preserve previous manifest', self.manifest()['error'])

    def test_recovery_write_failure_restores_complete_native_pair_and_can_retry(self):
        self.assertEqual(self.pipeline.run_one(self.spec, 'Example', 'Tiny', 'baseline', True), 'success')
        original = {path.name:path.read_bytes() for path in self.evals.glob('*.json')}
        self.fail_publication()
        write_json = self.pipeline.write_json
        def fail_average(path, value):
            if path == self.evals/'avg.json':
                raise OSError('fixture publication write failure')
            return write_json(path, value)
        with mock.patch.object(self.pipeline, 'write_json', side_effect=fail_average):
            self.assertEqual(self.recover(), 'failed')
        self.assertEqual({path.name:path.read_bytes() for path in self.evals.glob('*.json')}, original)
        self.assertEqual(self.manifest()['execution_phase'], 'publication')
        self.assertFalse(self.manifest()['canonical_metrics_committed'])
        self.assertEqual(self.recover(run_id='retry-write'), 'success')


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.method = self.root/'EasyTSAD/Methods/Example'
        project = self.method/'environment'
        project.mkdir(parents=True)
        (project/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.0.0"\nrequires-python=">=3.11"\ndependencies=[]\n')
        self.registry = self.root/'envs/registry.toml'
        self.registry.parent.mkdir(parents=True)
        (self.registry.parent/'shared').mkdir()
        (self.registry.parent/'shared/pyproject.toml').write_text('[project]\nname="shared"\n')
        self.registry.write_text('# preserve shared assignment\n[envs.shared]\npython="3.11"\nmethods=["Example","Other"]\n')
        self.bin = self.root/'bin'
        self.bin.mkdir()
        uv = self.bin/'uv'
        uv.write_text('''#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
project=Path(sys.argv[sys.argv.index('--project')+1])
Path(os.environ['FAKE_UV_RECORD']).write_text(json.dumps({'argv':sys.argv[1:],'venv':os.environ.get('UV_PROJECT_ENVIRONMENT'),'python':os.environ.get('UV_PYTHON_INSTALL_DIR'),'cache':os.environ.get('UV_CACHE_DIR'),'cwd':os.getcwd()}))
mode=os.environ.get('FAKE_UV_MODE','ok')
if mode=='sleep': time.sleep(120)
if mode=='fail': print('fixture dependency resolution failed',flush=True);sys.exit(7)
if mode=='quota': print('Disk quota exceeded (os error 122)',flush=True);sys.exit(2)
if mode!='nolock': (project/'uv.lock').write_text('# fixture lock')
(project/'.venv').mkdir(exist_ok=True)
for distribution,module in json.loads(os.environ.get('FAKE_UV_PACKAGES','{}')).items():
    if '--reinstall' in sys.argv or any(sys.argv[index:index+2]==['--reinstall-package',distribution] for index in range(len(sys.argv))):
        package=project/'.venv/lib/python3.11/site-packages'/module
        package.mkdir(parents=True,exist_ok=True)
        (package/'__init__.py').write_text('fixture=True\\n')
''')
        uv.chmod(0o755)
        self.processes = []
        self.task = SimpleNamespace(method='Example',method_dir=self.method)
        self.env = dict(os.environ,PATH=str(self.bin)+os.pathsep+os.environ.get('PATH',''),FAKE_UV_RECORD=str(self.root/'uv-record.json'))
        self.config = {'paths': {}}
        self.patch = mock.patch.multiple(environments,ROOT=self.root,load=lambda:SimpleNamespace(raw=self.config,network_env=lambda:dict(self.env)))
        self.patch.start()
    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()
    def declare(self, *dependencies, locked=()):
        project = self.method/'environment'
        (project/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.0.0"\n'
            'requires-python=">=3.11"\ndependencies='+json.dumps(dependencies)+'\n')
        if locked:
            (project/'uv.lock').write_text(''.join(
                '[[package]]\nname='+json.dumps(name)+'\nversion="1.0"\n' for name in locked))
    def distribution(self, name, *, top_level=None, records=()):
        directory = self.method/'environment/.venv/lib/python3.11/site-packages'/(
            name.replace('-', '_')+'-1.0.dist-info')
        directory.mkdir(parents=True)
        (directory/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: 1.0\n\n')
        if top_level is not None:
            (directory/'top_level.txt').write_text(top_level+'\n')
        if records:
            (directory/'RECORD').write_text(''.join(path+',,\n' for path in records))
        return directory
    def test_real_fake_uv_install_preserves_shared_registry_and_uses_method_venv(self):
        evidence = self.method/'reproduction/environment.json'
        evidence.parent.mkdir()
        evidence.write_text('{"evidence":[{"file":"upstream/requirements.txt","line":1}]}')
        original = evidence.read_bytes()
        result = environments.install(self.task,on_process=self.processes.append)
        self.assertEqual(evidence.read_bytes(),original)
        record = json.loads((self.root/'uv-record.json').read_text())
        self.assertEqual(record['argv'],['sync','--project',str(self.method/'environment')])
        self.assertEqual(record['venv'],str(self.method/'environment/.venv'))
        self.assertEqual(record['cwd'],str(self.root))
        self.assertEqual(record['python'],str(self.root/'runtime/python'))
        registry = tomllib.loads(self.registry.read_text())
        self.assertEqual(registry['envs']['shared']['methods'],['Other'])
        self.assertEqual(registry['envs']['method_Example']['methods'],['Example'])
        self.assertTrue((self.root/result['lock']).exists())
        self.assertTrue((self.method/'reproduction/environment-install.json').exists())
        self.assertEqual(result['action']['mode'],'sync')
        self.assertEqual(result['action']['packages'],[])
        self.assertEqual(json.loads((self.method/'reproduction/environment-install.json').read_text()),result)
        self.assertEqual([p['status'] for p in self.processes],['running','finished'])
        self.assertIsNotNone(self.processes[0]['processIdentity'])
        # The actual CLI resolves the same project and .venv from this registry.
        module_spec = importlib.util.spec_from_file_location('fixture_tsad_cli',REPO/'scripts/tsad.py')
        cli = importlib.util.module_from_spec(module_spec)
        with mock.patch.dict(os.environ,{'TSAD_ROOT':str(self.root)}):
            module_spec.loader.exec_module(cli)
        reg = cli.load_registry()
        self.assertEqual(reg['method_env']['Example'],'method_Example')
        self.assertEqual(cli.env_venv('method_Example'),self.method/'environment/.venv')
        self.assertEqual(cli.env_vars('method_Example')['UV_PROJECT_ENVIRONMENT'],record['venv'])
    def test_metadata_resolves_distribution_and_targeted_uv_repairs_missing_files(self):
        self.declare('scikit-learn>=1')
        self.distribution('scikit-learn',records=('sklearn/__init__.py','sklearn/metrics/__init__.py'))
        package = self.method/'environment/.venv/lib/python3.11/site-packages/sklearn/__init__.py'
        self.env['FAKE_UV_PACKAGES'] = json.dumps({'scikit-learn':'sklearn'})
        environments.install(self.task)
        self.assertFalse(package.exists())
        evidence = json.dumps({'error':"Traceback (most recent call last):\n"
                                      "ModuleNotFoundError: No module named 'sklearn.metrics'"})
        plan = environments.plan_recovery(self.task,evidence)
        self.assertEqual(set(plan),{'mode','packages','missing_import','reason'})
        self.assertEqual(plan['mode'],'reinstall')
        self.assertEqual(plan['packages'],['scikit-learn'])
        self.assertEqual(plan['missing_import'],'sklearn')
        result = environments.install(self.task,recovery=plan,on_process=self.processes.append)
        self.assertTrue(package.exists())
        record = json.loads((self.root/'uv-record.json').read_text())
        self.assertEqual(record['argv'],['sync','--project',str(self.method/'environment'),
                                        '--reinstall-package','scikit-learn'])
        self.assertEqual(result['action'],dict(plan,requested_mode='reinstall'))
        self.assertEqual(self.processes[0]['action'],result['action'])
        self.assertEqual(self.processes[-1]['action'],result['action'])
        self.assertEqual(json.loads((self.method/'reproduction/environment-install.json').read_text()),result)
        self.assertIn(json.dumps(result['action']),
                      (self.method/'reproduction/environment-install.log').read_text())
    def test_any_failed_targeted_repair_escalates_to_full_reinstall_once(self):
        self.declare('matplotlib','yacs')
        self.distribution('matplotlib',top_level='matplotlib')
        self.distribution('yacs',top_level='yacs')
        first = environments.plan_recovery(self.task,ModuleNotFoundError(name='matplotlib'))
        self.assertEqual(first['mode'],'reinstall')
        for missing in ('matplotlib','yacs'):
            with self.subTest(missing=missing):
                previous = dict(first,returnStage='IMPLEMENT')
                plan = environments.plan_recovery(self.task,ModuleNotFoundError(name=missing),previous=previous)
                self.assertEqual(previous,dict(first,returnStage='IMPLEMENT'))
                self.assertEqual(plan['mode'],'reinstall_all')
                self.assertEqual(plan['packages'],[])
                self.assertNotIn('returnStage',plan)
                final = environments.plan_recovery(self.task,ModuleNotFoundError(name=missing),previous=plan)
                self.assertEqual(final['mode'],'research')
                self.assertIn('full reinstall',final['reason'])
        plan = environments.plan_recovery(self.task,'ImportError: still broken',previous=first)
        self.assertEqual(plan['mode'],'reinstall_all')
        venv = self.method/'environment/.venv'
        sentinel = venv/'keep-interpreter'
        sentinel.write_text('do not recreate the venv')
        shared = self.root/'envs/shared/.venv'
        shared.mkdir()
        (shared/'keep').write_text('shared environment')
        self.env['FAKE_UV_PACKAGES'] = json.dumps({'matplotlib':'matplotlib','yacs':'yacs'})
        result = environments.install(self.task,recovery=plan,on_process=self.processes.append)
        record = json.loads((self.root/'uv-record.json').read_text())
        self.assertEqual(record['argv'],['sync','--project',str(self.method/'environment'),'--reinstall'])
        self.assertEqual(result['action']['mode'],'reinstall_all')
        self.assertEqual(sentinel.read_text(),'do not recreate the venv')
        self.assertEqual((shared/'keep').read_text(),'shared environment')
        for module in ('matplotlib','yacs'):
            self.assertTrue((venv/'lib/python3.11/site-packages'/module/'__init__.py').exists())
    def test_rebuild_alias_records_reinstall_all_and_research_runs_normal_sync(self):
        for requested, actual, flags in (('rebuild','reinstall_all',['--reinstall']),
                                          ('research','sync',[])):
            with self.subTest(requested=requested):
                result = environments.install(self.task,recovery={
                    'mode':requested,'packages':[],'reason':'fixture research completed',
                    'returnStage':'IMPLEMENT'})
                record = json.loads((self.root/'uv-record.json').read_text())
                self.assertEqual(record['argv'],['sync','--project',str(self.method/'environment')]+flags)
                self.assertEqual(result['action']['mode'],actual)
                self.assertEqual(result['action']['requested_mode'],requested)
                self.assertNotIn('returnStage',result['action'])
    def test_top_level_and_record_map_aliases_and_locked_transitive_dependencies(self):
        self.declare('parent',locked=('parent','PyYAML','Pillow'))
        self.distribution('PyYAML',top_level='yaml')
        self.distribution('Pillow',records=('PIL/Image.py',))
        for module, distribution in (('yaml','pyyaml'),('PIL.Image','pillow')):
            with self.subTest(module=module):
                error = ModuleNotFoundError("No module named 'misleading'",name=module)
                plan = environments.plan_recovery(self.task,error)
                self.assertEqual(plan['mode'],'reinstall')
                self.assertEqual(plan['packages'],[distribution])
                self.assertEqual(plan['missing_import'],module.split('.')[0])
    def test_missing_undeclared_dependency_requires_research(self):
        self.declare('numpy')
        self.distribution('numpy',top_level='numpy')
        for module in ('unknown_dependency','yaml'):
            with self.subTest(module=module):
                if module == 'yaml':
                    self.distribution('PyYAML',top_level='yaml')
                plan = environments.plan_recovery(self.task,"ModuleNotFoundError: No module named '"+module+"'")
                self.assertEqual(plan['mode'],'research')
                self.assertEqual(plan['packages'],[])
                self.assertIn('depen',plan['reason'])
    def test_uncertain_or_missing_metadata_uses_full_reinstall_without_guessing(self):
        self.declare('scikit-learn')
        plan = environments.plan_recovery(self.task,"ModuleNotFoundError: No module named 'sklearn'")
        self.assertEqual(plan['mode'],'research')
        self.assertEqual(plan['packages'],[])
        plan = environments.plan_recovery(self.task,"ModuleNotFoundError: No module named 'scikit_learn'")
        self.assertEqual(plan['mode'],'reinstall_all')
        self.assertEqual(plan['packages'],[])
        self.assertIn('without guessing',plan['reason'])
        self.distribution('scikit-learn',top_level='shared_namespace')
        self.distribution('another-owner',records=('shared_namespace/other.py',))
        plan = environments.plan_recovery(self.task,ModuleNotFoundError(name='shared_namespace'))
        self.assertEqual(plan['mode'],'reinstall_all')
        self.assertEqual(plan['packages'],[])
        final = environments.plan_recovery(self.task,ModuleNotFoundError(name='sklearn'),previous=plan)
        self.assertEqual(final['mode'],'research')
    def test_uninstalled_platform_and_optional_dependencies_do_not_imply_broken_venv(self):
        self.declare('numpy',locked=('numpy','windows-only','optional-package','dev-only'))
        self.distribution('numpy',top_level='numpy')
        with (self.method/'environment/pyproject.toml').open('a') as stream:
            stream.write('[project.optional-dependencies]\nplot=["optional-package"]\n'
                         '[dependency-groups]\ndev=["dev-only"]\n')
        plan = environments.plan_recovery(self.task,ModuleNotFoundError(name='unknown_dependency'))
        self.assertEqual(plan['mode'],'research')
        self.assertEqual(plan['packages'],[])
        self.assertIn('missing dependency',plan['reason'])
    def test_planner_reads_only_dist_info_and_does_not_import_or_modify_packages(self):
        self.declare('actual-distribution')
        self.distribution('actual-distribution',records=('missing.py',))
        site = self.method/'environment/.venv/lib/python3.11/site-packages'
        ordinary = site/'ignore_this_package'
        ordinary.mkdir()
        (ordinary/'METADATA').write_text('Name: wrong-distribution\n')
        (ordinary/'top_level.txt').write_text('missing\n')
        (ordinary/'__init__.py').write_text('raise RuntimeError("must not import")\n')
        original_glob = Path.glob
        patterns = []
        def metadata_only(path, pattern):
            patterns.append(pattern)
            self.assertTrue(pattern.endswith('/*.dist-info'))
            self.assertNotIn('**',pattern)
            return original_glob(path,pattern)
        before = self.registry.read_bytes()
        with mock.patch.object(Path,'glob',metadata_only), \
             mock.patch.object(environments.subprocess,'Popen') as spawn:
            plan = environments.plan_recovery(self.task,ModuleNotFoundError(name='missing'))
        self.assertEqual(plan['packages'],['actual-distribution'])
        self.assertTrue(patterns)
        spawn.assert_not_called()
        self.assertEqual(self.registry.read_bytes(),before)
        self.assertFalse((self.method/'reproduction').exists())
    def test_invalid_recovery_is_rejected_before_registry_or_process_changes(self):
        before = self.registry.read_bytes()
        actions = ['reinstall',{}, {'mode':'unknown'}, {'mode':[]}, {'mode':'reinstall','packages':[]},
                   {'mode':'reinstall','packages':['--all']},
                   {'mode':'reinstall','packages':['matplotlib>=3']},
                   {'mode':'reinstall','packages':'matplotlib'},
                   {'mode':'reinstall_all','packages':['numpy']}]
        for action in actions:
            with self.subTest(action=action), mock.patch.object(environments.subprocess,'Popen') as spawn:
                with self.assertRaises(HarnessError):
                    environments.install(self.task,recovery=action)
                spawn.assert_not_called()
        self.assertEqual(self.registry.read_bytes(),before)
    def test_targeted_packages_are_canonicalized_and_deduplicated(self):
        result = environments.install(self.task,recovery={
            'mode':'reinstall','packages':['Scikit_Learn','numpy','scikit-learn']})
        self.assertEqual(result['action']['packages'],['numpy','scikit-learn'])
        record = json.loads((self.root/'uv-record.json').read_text())
        self.assertEqual(record['argv'][-4:],['--reinstall-package','numpy','--reinstall-package','scikit-learn'])
    def test_full_repair_rejects_venv_symlinks_before_registry_changes(self):
        shared = self.root/'envs/shared/.venv'
        shared.mkdir()
        sentinel = shared/'keep'
        sentinel.write_text('untouched')
        (self.method/'environment/.venv').symlink_to(shared,target_is_directory=True)
        before = self.registry.read_bytes()
        with mock.patch.object(environments.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(HarnessError,'not a symlink'):
                environments.install(self.task,recovery={'mode':'reinstall_all'})
            spawn.assert_not_called()
        self.assertEqual(self.registry.read_bytes(),before)
        self.assertEqual(sentinel.read_text(),'untouched')
    def test_repair_failure_ignores_old_quota_and_action_rationale_and_clears_stale_success(self):
        environments.install(self.task)
        log = self.method/'reproduction/environment-install.log'
        with log.open('a') as stream:
            stream.write('Earlier attempt: Disk quota exceeded\n')
        self.env['FAKE_UV_MODE'] = 'fail'
        with self.assertRaises(environments.EnvironmentInstallError) as failure:
            environments.install(self.task,on_process=self.processes.append,recovery={
                'mode':'reinstall_all','reason':'Repair after Disk quota exceeded'})
        self.assertNotIn('Disk quota exceeded',str(failure.exception))
        self.assertFalse((self.method/'reproduction/environment-install.json').exists())
        self.assertEqual(self.processes[-1]['returncode'],7)
        self.assertEqual(self.processes[-1]['action']['mode'],'reinstall_all')
        self.assertIn('Repair after Disk quota exceeded',log.read_text())
    def test_repair_quota_failure_and_missing_lock_never_publish_success(self):
        for mode, exception in (('quota',environments.EnvironmentStorageError),('nolock',HarnessError)):
            with self.subTest(mode=mode):
                self.env['FAKE_UV_MODE'] = mode
                with self.assertRaises(exception):
                    environments.install(self.task,on_process=self.processes.append,recovery={
                        'mode':'reinstall','packages':['matplotlib']})
                self.assertFalse((self.method/'reproduction/environment-install.json').exists())
                self.assertEqual(self.processes[-1]['status'],'finished')
                self.assertEqual(self.processes[-1]['action']['mode'],'reinstall')
    def test_repair_cancellation_reaps_child_and_clears_previous_success(self):
        environments.install(self.task)
        self.env['FAKE_UV_MODE'] = 'sleep'
        cancel = SimpleNamespace(is_set=lambda:True)
        with self.assertRaisesRegex(HarnessError,'cancelled'):
            environments.install(self.task,on_process=self.processes.append,cancel=cancel,
                                 recovery={'mode':'reinstall_all'})
        with self.assertRaises(ProcessLookupError):
            os.kill(self.processes[0]['pid'],0)
        self.assertEqual(self.processes[-1]['status'],'finished')
        self.assertFalse((self.method/'reproduction/environment-install.json').exists())
    def test_escaped_double_quote_evidence_and_damaged_metadata_do_not_guess(self):
        self.declare('PyYAML')
        directory = self.distribution('PyYAML',top_level='yaml')
        error = json.dumps({'error':'ModuleNotFoundError: No module named "yaml.loader"'})
        plan = environments.plan_recovery(self.task,error)
        self.assertEqual(plan['packages'],['pyyaml'])
        (directory/'METADATA').unlink()
        plan = environments.plan_recovery(self.task,error)
        self.assertEqual(plan['mode'],'reinstall_all')
        self.assertEqual(plan['packages'],[])
        (self.method/'environment/uv.lock').write_text('invalid [toml')
        plan = environments.plan_recovery(self.task,error)
        self.assertEqual(plan['mode'],'reinstall_all')
        self.assertIn('declarations',plan['reason'])
    def test_install_failure_reports_uv_output_and_no_success_metadata(self):
        log = self.method/'reproduction/environment-install.log'
        log.parent.mkdir()
        log.write_text('Previous attempt: Disk quota exceeded (os error 122)\n')
        self.env['FAKE_UV_MODE']='fail'
        with self.assertRaisesRegex(HarnessError,'fixture dependency resolution failed') as failure:
            environments.install(self.task,on_process=self.processes.append)
        self.assertNotIsInstance(failure.exception,environments.EnvironmentStorageError)
        self.assertFalse((self.method/'reproduction/environment-install.json').exists())
        self.assertEqual(self.processes[-1]['returncode'],7)
    def test_external_cache_and_storage_error_keep_method_environment_local(self):
        self.config['paths']['uv_cache_dir'] = str(self.root/'local-disk/cache')
        self.env['FAKE_UV_MODE'] = 'quota'
        with self.assertRaises(environments.EnvironmentStorageError):
            environments.install(self.task,on_process=self.processes.append)
        record = json.loads((self.root/'uv-record.json').read_text())
        self.assertEqual(record['cache'],str(self.root/'local-disk/cache'))
        self.assertEqual(record['venv'],str(self.method/'environment/.venv'))
    def test_cancel_reaps_actual_child_without_installing(self):
        self.env['FAKE_UV_MODE']='sleep'
        with self.assertRaisesRegex(HarnessError,'cancelled'):
            environments.install(self.task,on_process=self.processes.append,cancel=lambda:True)
        pid = self.processes[0]['pid']
        with self.assertRaises(ProcessLookupError):
            os.kill(pid,0)
        self.assertFalse((self.method/'reproduction/environment-install.json').exists())
    def test_observer_failure_cleans_up_child(self):
        self.env['FAKE_UV_MODE']='sleep'
        def fail(record):
            self.processes.append(record)
            raise RuntimeError('fixture observer failed')
        with self.assertRaisesRegex(RuntimeError,'fixture observer failed'):
            environments.install(self.task,on_process=fail)
        with self.assertRaises(ProcessLookupError):
            os.kill(self.processes[0]['pid'],0)
    def test_missing_python_declaration_is_rejected_before_registry_change(self):
        project = self.method/'environment/pyproject.toml'
        project.write_text('[project]\nname="fixture"\ndependencies=[]\n')
        before = self.registry.read_bytes()
        with self.assertRaisesRegex(HarnessError,'requires-python'):
            environments.register(self.task)
        self.assertEqual(self.registry.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
