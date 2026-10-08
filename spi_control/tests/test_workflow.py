import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from layout import StartPageState, STOP_BTN
import start_page as ui


class WorkflowTests(unittest.TestCase):
    def test_execution_command_binds_exact_path_and_never_resolves(self):
        command=ui.vision_command(Path('python'),True,Path('plans/exact.json'),Path('session'),'reviewed-id')
        self.assertIn('--execute-approved',command)
        self.assertIn('plans\\exact.json' if sys.platform=='win32' else 'plans/exact.json',command)
        self.assertNotIn('--auto',command)
        self.assertNotIn('--execute',command)
        self.assertEqual(command[command.index('--expected-plan-id')+1],'reviewed-id')
        with self.assertRaises(ValueError):ui.vision_command(Path('python'),True)

    def test_unconfirmed_stop_is_never_reported_as_stopped(self):
        job=Mock()
        job.status.return_value={'stage':'failed','stop_confirmed':False}
        self.assertIn('未确认',ui.stop_workflow_job(job,True))
        job.close.assert_called_once()
        job.status.return_value={'stage':'failed','stop_confirmed':True}
        self.assertEqual(ui.stop_workflow_job(job,True),'停止已确认，本次方案失效')

    def test_unreadable_preview_blocks_confirmation(self):
        s=StartPageState();s.workflow_stage='preview';s.plan_ready=True
        s.review_image_path=Path('missing-preview.png')
        colors=dict.fromkeys(('scr','text','danger','primary','disabled','card'),0)
        with patch.object(ui,'paint_button'),patch('PIL.Image.open',side_effect=OSError('unreadable')):
            ui.paint_workflow(Mock(),Mock(),s,colors)
        self.assertEqual(s.workflow_stage,'failed');self.assertFalse(s.plan_ready)
        s.armed=True
        self.assertNotEqual(s.on_points([(450,350)]),'workflow_confirm')

    def test_busy_confirmation_only_at_explicit_wait_stages(self):
        s=StartPageState();s.busy=True
        for stage,expected in [('await_origin','workflow_confirm'),('running',None),
                               ('await_empty_accept','workflow_confirm'),('verifying',None),
                               ('await_physical_accept','workflow_confirm')]:
            s.workflow_stage=stage;s.armed=True
            self.assertEqual(s.on_points([(450,350)]),expected)
            s.armed=True
            self.assertEqual(s.on_points([(STOP_BTN[0]+1,STOP_BTN[1]+1)]),'stop')

    def test_confirmation_message_is_bound_to_nonce_plan_and_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            job=ui.VisionJob.__new__(ui.VisionJob);job.session_dir=Path(directory)
            status={'stage':'await_origin','session_id':'connection','plan_id':'plan','sequence':2}
            (job.session_dir/'status.json').write_text(json.dumps(status),encoding='utf8')
            job.confirm()
            command=json.loads((job.session_dir/'command.json').read_text('utf8'))
            self.assertEqual(command,{'action':'confirm','session_id':'connection','plan_id':'plan','sequence':2})
            status['stage']='running'
            (job.session_dir/'status.json').write_text(json.dumps(status),encoding='utf8')
            with self.assertRaises(RuntimeError):job.confirm()
