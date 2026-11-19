import io
import json
from pathlib import Path
import queue
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from desktop import trainer


class Value:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def app_stub(**values):
    app = SimpleNamespace(proc=None, output=None, active_run=None, output_kind='bbox', pending_job=None,
                          job_id=0, close_when_finished=False, messages=queue.Queue(), history=[], metric_timestamp=None,
                          progress=Value(), metric_info=Value(), gpu_info=Value(), logs=[], timers=[])
    app.append = app.logs.append
    app.after = lambda *args: app.timers.append(args)
    app.draw_chart = lambda: None
    app.read_json = trainer.Trainer.read_json
    app.run_kind = trainer.Trainer.run_kind
    for name in ('busy', 'background', 'set_output', 'launch', 'drain', 'update_metrics', 'stop_train', 'pose_values'):
        setattr(app, name, MethodType(getattr(trainer.Trainer, name), app))
    for name, value in [('dataset', ''), ('pose_dataset', ''), ('pose_checkpoint', ''), ('pose_scale', 'n'),
                        ('pose_epochs', 25), ('pose_batch', 2), ('pose_imgsz', '480'), ('pose_device', 'auto'),
                        ('pose_eval_split', 'test')]:
        setattr(app, name, Value(value))
    for key, value in values.items():
        setattr(app, key, value)
    return app


class DeferredThread:
    instances = []

    def __init__(self, target, daemon):
        self.target = target
        self.started = False
        self.instances.append(self)

    def start(self):
        self.started = True


class PoseDesktopTests(unittest.TestCase):
    def setUp(self):
        self.dialogs = []
        for method in ('showerror', 'showinfo'):
            dialog = patch.object(trainer.messagebox, method, side_effect=lambda *args: self.dialogs.append(args))
            dialog.start()
            self.addCleanup(dialog.stop)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'pose-data'
        self.dataset.mkdir()
        (self.dataset / 'pose.yaml').write_text('{}', encoding='utf-8')
        self.initializer = self.root / 'initializer.pt'
        self.initializer.write_bytes(b'local initializer')
        self.run = self.root / 'data' / 'training' / 'pose-run'
        self.run.mkdir(parents=True)
        DeferredThread.instances.clear()

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def prepare_resume(self):
        weights = self.run / 'weights'
        weights.mkdir()
        for name in ('last.pt', 'resume.pt', 'epoch2.pt'):
            (weights / name).write_bytes(b'resume fixture')
        config = dict(dataset=str(self.dataset), checkpoint=str(self.initializer), task='pose', scale='s',
                      epochs=30, batch=3, imgsz=640, device='cpu')
        self.write_json(self.run / 'pose-config.json', config)
        return weights, config

    def prepare_evaluation(self):
        weights, config = self.prepare_resume()
        checkpoint = weights / 'best.pt'
        checkpoint.write_bytes(b'evaluation fixture only')
        self.write_json(self.run / 'pose-result.json', {'status': 'completed', 'checkpoint': str(checkpoint),
                                                      'validation_passed': False, 'split': 'val',
                                                      'metrics': {'metrics/mAP50(P)': .61}})
        return checkpoint, config

    def test_evaluate_defaults_to_test_and_uses_selected_run_config_without_reselecting(self):
        checkpoint, config = self.prepare_evaluation()
        commands = []
        app = app_stub(output=self.run, output_kind='pose', launch=lambda args, **options: commands.append((args, options)) or True)
        app.pose_batch.set(16)
        original_report = {'status': 'completed', 'split': 'test', 'metrics': {'metrics/mAP50(P)': .15}}
        self.write_json(self.run / 'pose-evaluation.json', original_report)
        trainer.Trainer.evaluate_pose(app)
        args, options = commands.pop()
        self.assertEqual(args[:3], ['-m', 'ml.train_pose', 'evaluate'])
        flags = dict(zip(args[3::2], args[4::2]))
        self.assertEqual(flags['--data'], str(self.dataset / 'pose.yaml'))
        self.assertEqual(flags['--checkpoint'], str(checkpoint))
        self.assertEqual(flags['--split'], 'test')
        self.assertEqual(flags['--batch'], str(config['batch']))
        self.assertEqual(flags['--imgsz'], str(config['imgsz']))
        self.assertEqual(flags['--device'], config['device'])
        output = Path(flags['--output'])
        self.assertEqual(output.parent, self.run / 'evaluations')
        self.assertFalse(output.exists())
        self.assertNotIn('output', options)
        output.mkdir(parents=True)
        self.write_json(output / 'pose-evaluation.json', {'status': 'completed', 'split': 'test', 'metrics': {'metrics/mAP50(P)': .42}})
        selected = self.root / 'new-selection'
        app.set_output(selected, 'bbox')
        options['completed']()
        self.assertEqual(app.output, selected)
        self.assertEqual(app.output_kind, 'bbox')
        self.assertEqual(app.read_json(self.run / 'pose-evaluation.json'), original_report)

    def test_evaluation_success_refreshes_only_completed_real_report(self):
        self.prepare_evaluation()
        process = SimpleNamespace(poll=lambda: None, stdout=io.StringIO('measured fixture only\n'), wait=lambda: 0)
        app = app_stub(output=self.run, output_kind='pose')
        with patch.object(trainer.subprocess, 'Popen', return_value=process) as child, patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.evaluate_pose(app)
        command = child.call_args.args[0]
        output = Path(command[command.index('--output') + 1])
        output.mkdir(parents=True)
        self.write_json(output / 'pose-evaluation.json', {'status': 'completed', 'split': 'test',
                        'validation_source_kinds': ['synthetic_catalog_scene'], 'metrics': {'metrics/mAP50(P)': .42}})
        DeferredThread.instances[-1].target()
        app.drain()
        self.assertIsNone(app.proc)
        self.assertEqual(app.output, self.run)
        self.assertIn('Held-out test (test)', app.metric_info.get())
        self.assertIn('mAP50(P): 0.420', app.metric_info.get())
        self.assertIn('Gate validation: Không đạt', app.metric_info.get())
        self.assertTrue(any(str(output / 'pose-evaluation.json') in log for log in app.logs))

    def test_evaluate_validation_choice_uses_unique_output_and_rejects_invalid_configuration(self):
        self.prepare_evaluation()
        commands = []
        app = app_stub(output=self.run, output_kind='pose', launch=lambda args, **options: commands.append(args) or True)
        app.pose_eval_split.set('val')
        trainer.Trainer.evaluate_pose(app)
        trainer.Trainer.evaluate_pose(app)
        self.assertEqual(commands[0][commands[0].index('--split') + 1], 'val')
        self.assertNotEqual(commands[0][commands[0].index('--output') + 1], commands[1][commands[1].index('--output') + 1])
        commands.clear()
        for changes in ({'dataset': 'relative'}, {'batch': 0}, {'imgsz': 512}, {'device': 'cuda'}):
            with self.subTest(changes=changes):
                self.write_json(self.run / 'pose-config.json', dict(dataset=str(self.dataset), batch=3, imgsz=640, device='cpu') | changes)
                trainer.Trainer.evaluate_pose(app)
                self.assertEqual(commands, [])
        app.pose_eval_split.set('train')
        trainer.Trainer.evaluate_pose(app)
        self.assertEqual(commands, [])

    def test_evaluate_rejects_missing_run_bbox_busy_and_escaping_checkpoint(self):
        self.prepare_evaluation()
        commands = []
        for values in (dict(), dict(output=self.run), dict(output=self.run, output_kind='pose', pending_job=4),
                       dict(output=self.run, output_kind='pose', proc=SimpleNamespace(poll=lambda: None))):
            with self.subTest(values=values):
                app = app_stub(**values, launch=lambda *args, **options: commands.append(args))
                trainer.Trainer.evaluate_pose(app)
                self.assertEqual(commands, [])
        self.write_json(self.run / 'pose-result.json', {'status': 'completed', 'checkpoint': str(self.initializer)})
        app = app_stub(output=self.run, output_kind='pose', launch=lambda *args, **options: commands.append(args))
        trainer.Trainer.evaluate_pose(app)
        self.assertEqual(commands, [])
        self.assertTrue(self.dialogs)

    def test_evaluation_child_failure_releases_worker_and_preserves_selected_run(self):
        self.prepare_evaluation()
        process = SimpleNamespace(poll=lambda: None, stdout=io.StringIO('evaluation failed\n'), wait=lambda: 1)
        app = app_stub(output=self.run, output_kind='pose')
        with patch.object(trainer.subprocess, 'Popen', return_value=process), patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.evaluate_pose(app)
        self.assertIs(app.proc, process)
        self.assertIsNone(app.active_run)
        self.assertFalse((self.run / 'pose-evaluation.json').exists())
        DeferredThread.instances[-1].target()
        app.drain()
        self.assertIsNone(app.proc)
        self.assertIsNone(app.active_run)
        self.assertEqual(app.output, self.run)
        self.assertIn('evaluation failed\n', app.logs)
        self.assertTrue(app.timers)

    def test_evaluation_missing_report_is_error_and_completion_callback_error_keeps_drain_alive(self):
        self.prepare_evaluation()
        process = SimpleNamespace(poll=lambda: None, stdout=io.StringIO(''), wait=lambda: 0)
        app = app_stub(output=self.run, output_kind='pose')
        with patch.object(trainer.subprocess, 'Popen', return_value=process), patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.evaluate_pose(app)
        DeferredThread.instances[-1].target()
        app.drain()
        self.assertIsNone(app.proc)
        self.assertTrue(self.dialogs)
        self.assertTrue(app.timers)
        self.assertFalse((self.run / 'pose-evaluation.json').exists())
        def failed_callback():
            raise OSError('report became unavailable')
        app.proc = process
        app.messages.put(('complete', (app.job_id, process, 0, failed_callback)))
        app.timers.clear()
        app.drain()
        self.assertIsNone(app.proc)
        self.assertTrue(app.timers)

    def test_metrics_show_saved_heldout_separately_without_replacing_validation_gate(self):
        self.prepare_evaluation()
        self.write_json(self.run / 'pose-evaluation.json', {'status': 'completed', 'split': 'test',
                        'split_role': 'held_out_test', 'validation_source_kinds': ['synthetic_catalog_scene'],
                        'validation_passed': True, 'metrics': {'metrics/mAP50(P)': .42}, 'real_photo_validated': False})
        app = app_stub(output=self.run, output_kind='pose')
        app.update_metrics()
        info = app.metric_info.get()
        self.assertIn('Validation (val)', info)
        self.assertIn('Held-out test (test)', info)
        self.assertIn('mAP50(P): 0.610', info)
        self.assertIn('mAP50(P): 0.420', info)
        self.assertIn('Gate validation: Kh\u00f4ng \u0111\u1ea1t', info)
        self.assertIn('Ngu\u1ed3n: t\u1ed5ng h\u1ee3p', info)
        self.assertIn('ch\u01b0a x\u00e1c nh\u1eadn camera th\u1eadt', info)

    def test_metrics_read_newest_separate_evaluation_and_detect_standalone_pose_report(self):
        self.prepare_evaluation()
        app = app_stub(output=self.run, output_kind='pose')
        app.update_metrics()
        self.assertNotIn('Held-out test (test)', app.metric_info.get())
        output = self.run / 'evaluations' / 'val-123'
        output.mkdir(parents=True)
        self.write_json(output / 'pose-evaluation.json', {'status': 'completed', 'split': 'val',
                        'validation_source_kinds': ['synthetic_catalog_scene'], 'metrics': {'metrics/mAP50(P)': .47}})
        app.update_metrics()
        self.assertIn('mAP50(P): 0.470', app.metric_info.get())
        self.assertIn('Validation (val)', app.metric_info.get())
        self.assertNotIn('Held-out test (test)', app.metric_info.get())
        self.assertEqual(trainer.Trainer.run_kind(output), 'pose')
        standalone = app_stub()
        standalone.set_output(output)
        standalone.update_metrics()
        self.assertIn('mAP50(P): 0.470', standalone.metric_info.get())
        self.assertNotIn('Gate validation', standalone.metric_info.get())
        self.write_json(output / 'pose-evaluation.json', {'status': 'failed', 'split': 'test', 'metrics': {'metrics/mAP50(P)': .99}})
        app.update_metrics()
        self.assertNotIn('mAP50(P): 0.990', app.metric_info.get())

    def test_train_command_uses_local_initializer_and_all_selected_values(self):
        command = trainer.pose_command(self.dataset, self.initializer, self.run, 's', 30, 3, 640, '0')
        self.assertEqual(command[:3], ['-m', 'ml.train_pose', 'train'])
        flags = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(flags, {'--data': str(self.dataset / 'pose.yaml'), '--checkpoint': str(self.initializer),
                                '--output': str(self.run), '--scale': 's', '--epochs': '30', '--batch': '3',
                                '--imgsz': '640', '--device': '0'})
        self.assertFalse((self.run / 'pose-config.json').exists())

    def test_command_rejects_missing_local_model_invalid_settings_and_yaml_as_folder(self):
        cases = [dict(checkpoint=self.root / 'yolo26n.pt'), dict(scale='m'), dict(epochs=0), dict(batch=17),
                 dict(imgsz=512), dict(device='cuda'), dict(dataset=self.dataset / 'pose.yaml')]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                trainer.pose_command(**dict(dataset=self.dataset, checkpoint=self.initializer, output=self.run) | changes)

    def test_resume_prefers_actual_resume_and_preserves_original_config(self):
        weights, original = self.prepare_resume()
        output, config, command = trainer.pose_resume_settings(weights / 'last.pt')
        self.assertEqual(output, self.run)
        self.assertEqual(config, original)
        self.assertEqual(command[command.index('--checkpoint') + 1], str(weights / 'resume.pt'))
        self.assertEqual(command[command.index('--data') + 1], str(self.dataset / 'pose.yaml'))
        self.assertEqual(command[command.index('--scale') + 1], 's')
        self.assertEqual(command[command.index('--epochs') + 1], '30')
        self.assertEqual(command[command.index('--batch') + 1], '3')
        self.assertEqual(command[command.index('--imgsz') + 1], '640')
        self.assertEqual(command[command.index('--device') + 1], 'cpu')
        self.assertEqual(command[-1], '--resume')
        self.assertEqual(json.loads((self.run / 'pose-config.json').read_text()), original)
        _, _, epoch_command = trainer.pose_resume_settings(weights / 'epoch2.pt')
        self.assertEqual(epoch_command[epoch_command.index('--checkpoint') + 1], str(weights / 'epoch2.pt'))

    def test_resume_rejects_relative_or_yaml_dataset_and_best_checkpoint(self):
        weights, config = self.prepare_resume()
        for dataset in ('relative-folder', str(self.dataset / 'pose.yaml')):
            with self.subTest(dataset=dataset):
                self.write_json(self.run / 'pose-config.json', {**config, 'dataset': dataset})
                with self.assertRaises(ValueError):
                    trainer.pose_resume_settings(weights / 'last.pt')
        (weights / 'best.pt').write_bytes(b'best is not resumable')
        with self.assertRaises(ValueError):
            trainer.pose_resume_settings(weights / 'best.pt')

    def test_prepare_and_reviewed_import_launch_child_commands(self):
        app = app_stub()
        commands = []
        app.launch = lambda args, **kwargs: commands.append((args, kwargs))
        with patch.object(trainer.filedialog, 'askdirectory', return_value=str(self.dataset)):
            trainer.Trainer.prepare_pose_scenes(app)
        args, options = commands.pop()
        self.assertEqual(args[:4], ['-m', 'ml.pose_dataset', 'prepare', str(self.dataset)])
        self.assertFalse(Path(args[4]).exists())
        options['completed']()
        self.assertEqual(app.pose_dataset.get(), args[4])
        manifest = self.root / 'reviewed.jsonl'
        with patch.object(trainer.filedialog, 'askopenfilename', return_value=str(manifest)):
            trainer.Trainer.import_pose_manifest(app)
        args, options = commands.pop()
        self.assertEqual(args[:3], ['-m', 'ml.prepare_pose', str(manifest)])
        options['completed']()
        self.assertEqual(app.pose_dataset.get(), args[3])

    def test_prepared_dataset_completion_does_not_replace_new_selection(self):
        app = app_stub()
        jobs = []
        app.launch = lambda *args, **options: jobs.append(options['completed'])
        with patch.object(trainer.filedialog, 'askdirectory', return_value=str(self.dataset)):
            trainer.Trainer.prepare_pose_scenes(app)
        app.pose_dataset.set('new-selection')
        jobs.pop()()
        self.assertEqual(app.pose_dataset.get(), 'new-selection')

    def test_start_pose_waits_for_validation_and_discards_changed_selection(self):
        app = app_stub()
        app.pose_dataset.set(str(self.dataset))
        app.pose_checkpoint.set(str(self.initializer))
        validated, launched = [], []
        app.pose_validation = lambda dataset: validated.append(dataset) or {'valid': True, 'counts': {'train': 1}}
        app.show_validation = lambda result: None
        app.launch = lambda *args, **kwargs: launched.append((args, kwargs))
        with patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.start_pose_train(app)
        self.assertEqual(validated, [])
        self.assertEqual(launched, [])
        self.assertTrue(DeferredThread.instances[-1].started)
        DeferredThread.instances[-1].target()
        app.pose_dataset.set(str(self.root / 'other'))
        app.drain()
        self.assertEqual(validated, [str(self.dataset)])
        self.assertEqual(launched, [])
        self.assertIsNone(app.pending_job)

    def test_launch_tracks_both_workers_and_stop_targets_worker_not_selected_result(self):
        for command in (['-m', 'ml.train_detection'], ['-m', 'ml.train_pose', 'train']):
            with self.subTest(command=command):
                process = SimpleNamespace(poll=lambda: None, stdout=io.StringIO('worker log\n'), wait=lambda: 0)
                app = app_stub()
                with patch.object(trainer.subprocess, 'Popen', return_value=process), patch.object(trainer.threading, 'Thread', DeferredThread):
                    self.assertTrue(app.launch(command, output=self.run))
                self.assertIs(app.proc, process)
                self.assertEqual(app.active_run, self.run)
                app.output = self.root / 'another-run'
                self.assertTrue(app.stop_train())
                self.assertTrue((self.run / 'STOP').is_file())
                self.assertFalse((app.output / 'STOP').exists())
                DeferredThread.instances[-1].target()
                app.drain()
                self.assertIsNone(app.proc)
                self.assertIsNone(app.active_run)
                self.assertIn('worker log\n', app.logs)

    def test_failed_launch_preserves_existing_process_and_run(self):
        previous = SimpleNamespace(poll=lambda: 1)
        app = app_stub(proc=previous, active_run=self.root / 'previous', output=self.root / 'selected')
        with patch.object(trainer.subprocess, 'Popen', side_effect=OSError('launch failed')), patch.object(trainer.messagebox, 'showerror'):
            self.assertFalse(app.launch(['-m', 'ml.train_pose', 'train'], output=self.run))
        self.assertIs(app.proc, previous)
        self.assertEqual(app.active_run, self.root / 'previous')
        self.assertEqual(app.output, self.root / 'selected')
        self.assertEqual(app.job_id, 0)

    def test_stale_worker_completion_cannot_clear_new_worker_or_call_callback(self):
        current, previous = object(), object()
        called = []
        app = app_stub(proc=current, active_run=self.run, job_id=2)
        app.messages.put(('complete', (1, previous, 0, lambda: called.append(True))))
        app.drain()
        self.assertIs(app.proc, current)
        self.assertEqual(app.active_run, self.run)
        self.assertEqual(called, [])

    def test_close_requests_epoch_stop_keeps_reader_alive_and_destroys_after_exit(self):
        process = SimpleNamespace(poll=lambda: None)
        destroyed, hidden = [], []
        app = app_stub(proc=process, active_run=self.run, job_id=3,
                       destroy=lambda: destroyed.append(True), withdraw=lambda: hidden.append(True))
        with patch.object(trainer.messagebox, 'askyesno', return_value=True):
            trainer.Trainer.close(app)
        self.assertTrue((self.run / 'STOP').is_file())
        self.assertEqual(destroyed, [])
        self.assertEqual(hidden, [True])
        self.assertTrue(app.close_when_finished)
        app.messages.put(('complete', (3, process, 0, None)))
        app.drain()
        self.assertEqual(destroyed, [True])
        self.assertEqual(app.timers, [])

    def test_log_read_failure_keeps_wait_and_complete_tracking(self):
        class BrokenStream:
            def __iter__(self):
                raise OSError('log pipe interrupted')
        process = SimpleNamespace(poll=lambda: None, stdout=BrokenStream(), wait=lambda: 2)
        app = app_stub()
        with patch.object(trainer.subprocess, 'Popen', return_value=process), patch.object(trainer.threading, 'Thread', DeferredThread):
            self.assertTrue(app.launch(['-m', 'ml.train_pose', 'train'], output=self.run))
        DeferredThread.instances[-1].target()
        app.drain()
        self.assertIsNone(app.proc)
        self.assertIsNone(app.active_run)
        self.assertTrue(any('log pipe interrupted' in line for line in app.logs))

    def test_pose_metrics_use_p_values_and_refresh_result_without_csv_change(self):
        self.write_json(self.run / 'pose-config.json', {'dataset': str(self.dataset)})
        self.write_json(self.dataset / 'dataset.json', {'kind': 'synthetic_only'})
        (self.run / 'results.csv').write_text('epoch,metrics/mAP50(B),metrics/mAP50(P),metrics/precision(P)\n1,.99,.62,.74\n', encoding='utf-8')
        app = app_stub(output=self.run, output_kind='pose')
        app.update_metrics()
        self.assertIn('mAP50(P): 0.620', app.metric_info.get())
        self.assertIn('Recall(P): chưa có', app.metric_info.get())
        self.assertNotIn('0.990', app.metric_info.get())
        timestamp = app.metric_timestamp
        self.write_json(self.run / 'pose-result.json', {'status': 'completed', 'validation_passed': True, 'split': 'test',
                                                      'metrics': {'metrics/mAP50(P)': .81}, 'real_photo_validated': False})
        app.update_metrics()
        self.assertEqual(app.metric_timestamp, timestamp)
        self.assertIn('mAP50(P): 0.810', app.metric_info.get())
        self.assertIn('Split: test', app.metric_info.get())
        self.assertIn('Gate validation: Đạt', app.metric_info.get())
        self.assertIn('Nguồn: tổng hợp', app.metric_info.get())
        self.assertIn('chưa xác nhận camera thật', app.metric_info.get())

    def test_starting_false_gate_is_unmeasured_and_nonfinite_metrics_are_not_values(self):
        self.write_json(self.run / 'pose-result.json', {'status': 'starting', 'validation_passed': False,
                                                      'metrics': {'metrics/mAP50(P)': float('nan')}})
        app = app_stub(output=self.run, output_kind='pose')
        app.update_metrics()
        self.assertIn('Gate validation: Chưa đánh giá', app.metric_info.get())
        self.assertIn('mAP50(P): chưa có', app.metric_info.get())

    def test_status_metrics_work_without_csv_and_select_run_resets_previous_history(self):
        self.write_json(self.run / 'pose-status.json', {'epoch': 1, 'epochs': 4, 'metrics': {'metrics/recall(P)': .51}})
        app = app_stub(history=[{'metrics/mAP50(B)': .99}])
        with patch.object(trainer.filedialog, 'askdirectory', return_value=str(self.run)):
            trainer.Trainer.select_run(app)
        self.assertEqual(app.output_kind, 'pose')
        self.assertEqual(app.history, [])
        self.assertIn('Recall(P): 0.510', app.metric_info.get())
        self.assertIn('mAP50(P): chưa có', app.metric_info.get())
        app.set_output(self.root)
        self.assertEqual(app.output_kind, 'bbox')

    def test_activation_is_deferred_and_dispatches_pose_experimental_gate(self):
        called = []
        fake_training = SimpleNamespace(Activate=lambda **kwargs: kwargs,
                                        activate=lambda *args: self.fail('bbox activation called'),
                                        activate_pose=lambda run_id, **kwargs: called.append((run_id, kwargs)) or {'active': run_id})
        app = app_stub(output=self.run, output_kind='pose')
        with patch.dict('sys.modules', {'cardlens_ai.training': fake_training}), patch.object(trainer, 'ROOT', self.root), patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.activate(app)
            self.assertEqual(called, [])
            DeferredThread.instances[-1].target()
        app.drain()
        self.assertEqual(called, [('pose-run', {'experimental': True})])

    def test_resume_handler_restores_ui_initializer_and_defers_validation(self):
        weights, config = self.prepare_resume()
        (self.run / 'STOP').touch()
        app = app_stub()
        launched = []
        app.pose_validation = lambda dataset: {'valid': True, 'counts': {}}
        app.launch = lambda args, **options: launched.append((args, options))
        with patch.object(trainer.filedialog, 'askopenfilename', return_value=str(weights / 'last.pt')), patch.object(trainer.threading, 'Thread', DeferredThread):
            trainer.Trainer.resume_pose(app)
        self.assertEqual(app.pose_dataset.get(), config['dataset'])
        self.assertEqual(app.pose_checkpoint.get(), config['checkpoint'])
        self.assertEqual(app.pose_scale.get(), 's')
        self.assertEqual(app.pose_device.get(), 'cpu')
        self.assertEqual(launched, [])
        self.assertTrue((self.run / 'STOP').exists())
        DeferredThread.instances[-1].target()
        app.drain()
        command, options = launched[0]
        self.assertEqual(options['output'], self.run)
        self.assertEqual(command[command.index('--checkpoint') + 1], str(weights / 'resume.pt'))
        self.assertEqual(command[-1], '--resume')

    def test_chart_uses_pose_values_and_omits_missing_series(self):
        class Chart:
            def __init__(self): self.lines, self.ovals, self.texts = [], [], []
            def delete(self, *args): pass
            def winfo_width(self): return 600
            def winfo_height(self): return 205
            def create_text(self, *args, **options): self.texts.append(options['text'])
            def create_line(self, *args, **options): self.lines.append((args, options))
            def create_oval(self, *args, **options): self.ovals.append((args, options))
        app = app_stub(output_kind='pose', chart=Chart(), history=[{'metrics/mAP50(P)': .4, 'metrics/mAP50(B)': .9}])
        trainer.Trainer.draw_chart(app)
        self.assertEqual(len(app.chart.ovals), 1)
        self.assertIn('mAP50(P)', app.chart.texts)
        self.assertNotIn('mAP50(B)', app.chart.texts)
        x1, y1, x2, y2 = app.chart.ovals[0][0]
        self.assertAlmostEqual((y1 + y2) / 2, 170 - .4 * 135)
        self.assertEqual(len(app.chart.lines), 5)

    def test_bbox_results_still_use_bbox_metrics_and_validation_gate(self):
        self.write_json(self.run / 'result.json', {'validation_passed': False, 'metrics': {'metrics/mAP50(B)': .67},
                                                 'dataset_kind': 'synthetic_only', 'status': 'completed'})
        app = app_stub(output=self.run, output_kind='bbox')
        app.update_metrics()
        self.assertIn('mAP50(B): 0.670', app.metric_info.get())
        self.assertIn('Gate validation: Không đạt', app.metric_info.get())
        self.assertNotIn('mAP50(P)', app.metric_info.get())

    def test_export_dispatches_pose_and_bbox_children(self):
        checkpoint = self.run / 'best.pt'
        checkpoint.write_bytes(b'pose')
        self.write_json(self.run / 'pose-result.json', {'checkpoint': str(checkpoint)})
        commands = []
        app = app_stub(output=self.run, output_kind='pose', launch=lambda args: commands.append(args))
        trainer.Trainer.export(app)
        self.assertEqual(commands.pop(), ['-m', 'ml.train_pose', 'export', '--checkpoint', str(checkpoint), '--output', str(self.run)])
        app.output_kind = 'bbox'
        trainer.Trainer.export(app)
        self.assertEqual(commands.pop(), ['-m', 'ml.export_detector', str(self.run)])


if __name__ == '__main__':
    unittest.main()