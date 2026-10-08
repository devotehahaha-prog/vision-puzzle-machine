import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ordinary_camera as ordinary
import start_page as ui
from layout import StartPageState,MODE_ORDINARY


class OrdinaryExecutionUITests(unittest.TestCase):
    def test_real_job_binds_exact_plan_and_inherits_stop_and_confirmation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ui.subprocess,'Popen') as popen:
            with patch.object(ui,'HERE',Path(folder)):
                job=ui.OrdinaryCameraJob(Path('python'),Path('puzzle_vision'),True,Path('bound.json'),'id')
                command=popen.call_args.args[0]
                self.assertIn('--execute-approved',command)
                self.assertNotIn('--prepare-execution',command)
                self.assertEqual(command[command.index('--expected-plan-id')+1],'id')
                self.assertIsInstance(job,ui.VisionJob)
                job.log.close()
                with self.assertRaises(ValueError):ui.OrdinaryCameraJob(Path('python'),Path('puzzle_vision'),True)

    def test_prepared_plan_shows_execution_review(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder);image=project/'assembly.png';image.write_bytes(b'placeholder')
            ordinary.atomic_json(project/'output_camera/plan.json',{
                'mode':'ordinary_camera','preview_ready':True,'ready_for_motion':False,
                'approval_state':'review_required','piece_count':4,'generated_at_epoch':time.time(),
                'schema_version':2,'puzzle_profile':'ordinary','content_sha256':'bound',
                'all_matches_ok':True,'all_paths_ok':True,'geometry_reasons':[],
                'assembly_path':str(image),'artifact_path':str(project/'bound.json'),'plan_id':'id'})
            s=StartPageState(default_mode=MODE_ORDINARY)
            ui.finish_ordinary_plan(s,project,0,0)
            self.assertTrue(s.plan_ready);self.assertEqual(s.workflow_stage,'preview')
            self.assertEqual(s.review_plan_path,project/'bound.json')

    def test_authorized_automatic_run_consumes_one_origin_confirmation(self):
        s=StartPageState(default_mode=MODE_ORDINARY)
        action,phase=ui.next_ordinary_auto_action('plan',s)
        self.assertEqual((action,phase),('start','review'))
        s.busy=True
        self.assertEqual(ui.next_ordinary_auto_action(phase,s),(None,'review'))
        s.busy=False;s.plan_ready=True;s.workflow_stage='preview'
        self.assertEqual(ui.next_ordinary_auto_action(phase,s),('workflow_confirm','origin'))
        s.busy=True;s.workflow_stage='await_origin'
        self.assertEqual(ui.next_ordinary_auto_action('origin',s),('workflow_confirm',''))
        self.assertEqual(ui.next_ordinary_auto_action('',s),(None,''))
        s.workflow_stage='await_physical_accept'
        self.assertEqual(ui.next_ordinary_auto_action('',s),(None,''))

    def test_failure_never_automatically_regenerates_or_retries_motion(self):
        s=StartPageState(default_mode=MODE_ORDINARY);s.workflow_stage='plan_failed'
        self.assertEqual(ui.next_ordinary_auto_action('review',s),(None,''))
        s.workflow_stage='failed'
        self.assertEqual(ui.next_ordinary_auto_action('origin',s),(None,''))
        with self.assertRaisesRegex(RuntimeError,'操作者确认'):ui.main(auto_ordinary=True)
