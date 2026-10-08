"""Regression checks for real-camera plan acceptance; no hardware is opened."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import main as vision


class StablePlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.output_patch = patch.object(vision, 'OUTPUT_DIR', self.output)
        self.output_patch.start()
        self.config = {'pixels_per_mm': 4, 'expected_piece_count': 3}
        self.frames = [np.full((2, 2, 3), i, np.uint8) for i in range(5)]

    def tearDown(self):
        self.output_patch.stop()
        self.directory.cleanup()

    def observation(self, index):
        pieces = [SimpleNamespace(center_mm=lambda ppm, j=j: (j*20., 200.),
                                  area_mm2=100., width_mm=10., height_mm=20.)
                  for j in range(3)]
        return (np.eye(3), self.frames[index], np.zeros((2, 2), np.uint8),
                pieces, [SimpleNamespace(frame=index,piece=p,
                         match=SimpleNamespace(target_pick_mm=(j*20.,50.),rotation_deg=0.))
                         for j,p in enumerate(pieces)])

    def capture(self):
        frames = iter(self.frames[1:])
        return SimpleNamespace(read=lambda: (True, next(frames)))

    def blocked_plan(self):
        return json.loads((self.output/'plan.json').read_text(encoding='utf-8'))

    def test_three_safe_frames_plus_one_unsafe_frame_do_not_pass(self):
        (self.output/'plan.json').write_text('{"ready_for_motion":true,"pieces":[1]}')
        outcomes = [RuntimeError('solve failed')] + [self.observation(i) for i in range(1,5)]
        def safety(paths, config):
            return ['boundary gap'] if paths[0].frame == 4 else []
        with patch.object(vision, '_auto_detect_and_solve', side_effect=outcomes), \
             patch.object(vision, 'motion_safety_reasons', side_effect=safety), \
             patch.object(vision, 'save_detection_diagnostics'), \
             patch.object(vision, 'save_outputs') as save:
            with self.assertRaisesRegex(RuntimeError, '仅3帧'):
                vision._stable_auto_detect_and_solve(self.capture(), self.frames[0], self.config, None)
        save.assert_not_called()
        self.assertFalse(self.blocked_plan()['ready_for_motion'])
        self.assertEqual(self.blocked_plan()['pieces'], [])

    def test_selected_frame_and_plan_are_saved_together(self):
        outcomes = [self.observation(i) for i in range(5)]
        def safety(paths, config):
            return ['bad last frame'] if paths[0].frame == 4 else []
        with patch.object(vision, '_auto_detect_and_solve', side_effect=outcomes), \
             patch.object(vision, 'motion_safety_reasons', side_effect=safety), \
             patch.object(vision, 'draw_preview', return_value=self.frames[0]), \
             patch.object(vision, 'save_outputs') as save:
            chosen = vision._stable_auto_detect_and_solve(self.capture(), self.frames[0], self.config, None)
        self.assertIs(chosen, outcomes[0])
        save.assert_called_once()
        self.assertIs(save.call_args.args[0], self.frames[0])
        self.assertIs(save.call_args.args[1], outcomes[0][1])
        self.assertIs(save.call_args.args[4], outcomes[0][4])

    def test_changed_source_positions_are_rejected_even_if_each_plan_is_safe(self):
        outcomes = [self.observation(i) for i in range(5)]
        for i, item in enumerate(outcomes):
            for j, piece in enumerate(item[3]):
                piece.center_mm = lambda ppm, i=i, j=j: (i*10.+j*20., 200.)
        with patch.object(vision, '_auto_detect_and_solve', side_effect=outcomes), \
             patch.object(vision, 'motion_safety_reasons', return_value=[]), \
             patch.object(vision, 'save_outputs') as save:
            with self.assertRaisesRegex(RuntimeError, '稳定阈值'):
                vision._stable_auto_detect_and_solve(self.capture(), self.frames[0], self.config, None)
        save.assert_not_called()
        self.assertFalse(self.blocked_plan()['ready_for_motion'])

    def test_capture_failure_invalidates_old_ready_plan(self):
        (self.output/'plan.json').write_text('{"ready_for_motion":true}')
        capture = SimpleNamespace(read=lambda: (False, None))
        with self.assertRaisesRegex(RuntimeError, '读取摄像头失败'):
            vision._stable_auto_detect_and_solve(capture, self.frames[0], self.config, None)
        self.assertFalse(self.blocked_plan()['ready_for_motion'])

    def test_still_sources_with_changing_target_rotations_are_rejected(self):
        outcomes = [self.observation(i) for i in range(5)]
        for i,item in enumerate(outcomes):
            item[4][0].match.rotation_deg = i*15.
        with patch.object(vision,'_auto_detect_and_solve',side_effect=outcomes), \
             patch.object(vision,'motion_safety_reasons',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'稳定阈值'):
                vision._stable_auto_detect_and_solve(self.capture(),self.frames[0],self.config,None)
        self.assertFalse(self.blocked_plan()['ready_for_motion'])

    def test_final_photo_detects_target_half(self):
        (self.output/'plan.json').write_text('{"piece_count":3}',encoding='utf8')
        config = {'source_region':'bottom','target_region':'top','paper_height_mm':297.}
        report = {'status':'pass','detected_piece_count':3,'expected_piece_count':3,
                  'matches':[],'missing_target_piece_ids':[]}
        with patch.object(vision,'warp_paper',return_value=self.frames[0]), \
             patch.object(vision,'detect_pieces',return_value=([],np.zeros((2,2),np.uint8))) as detect, \
             patch.object(vision,'match_final_layout',return_value=report), \
             patch.object(vision,'draw_final_verification',return_value=self.frames[0]), \
             patch.object(vision,'write_image'):
            vision.verify_final_frame(self.frames[0],np.eye(3),config,self.output/'plan.json')
        self.assertEqual(detect.call_args_list[0].args[1]['source_region'],'top')
        self.assertEqual(detect.call_args_list[1].args[1]['source_region'],'bottom')
        self.assertEqual(config['source_region'],'bottom')

    def test_chroma_segmentation_excludes_brightness_only_shadow(self):
        import cv2
        lab=np.full((100,100,3),(180,96,150),np.uint8)
        lab[20:40,20:40]=(130,96,150)
        lab[60:80,60:80]=(225,128,128)
        image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        cfg={'pixels_per_mm':1,'border_sample_mm':5,'background_distance_threshold':18}
        default=vision.make_background_inverse_mask(image,cfg)
        chroma=vision.make_background_inverse_mask(image,{**cfg,'background_distance_components':'ab'})
        self.assertEqual(default[30,30],255)
        self.assertEqual(chroma[30,30],0)
        self.assertEqual(chroma[70,70],255)

    def test_auto_frame_solving_defers_output_writes(self):
        with patch.object(vision, 'resolve_calibration_matrix', return_value=(np.eye(3), 'saved')), \
             patch.object(vision, 'warp_paper', return_value=self.frames[0]), \
             patch.object(vision, 'mask_in_paper', return_value=(None, None)), \
             patch.object(vision, 'process_paper', return_value=([], [])) as process:
            vision._auto_detect_and_solve(self.frames[0], self.config, None)
        self.assertFalse(process.call_args.kwargs['save'])

    def test_noninteractive_failure_returns_nonzero_and_releases_camera(self):
        capture = SimpleNamespace(release=unittest.mock.Mock())
        with patch.object(vision, 'load_calibration', return_value=None), \
             patch.object(vision, 'open_camera', return_value=capture), \
             patch.object(vision, 'capture_immediate_frame', return_value=(self.frames[0], 0)), \
             patch.object(vision, '_stable_auto_detect_and_solve', side_effect=RuntimeError('unsafe')), \
             patch.object(vision.cv2, 'destroyAllWindows'):
            with self.assertRaisesRegex(RuntimeError, '未生成通过'):
                vision.run_auto({'auto_max_retries':1, 'auto_fallback_enabled':False},
                                None, None, True, True)
        capture.release.assert_called_once()
        self.assertFalse(self.blocked_plan()['ready_for_motion'])

    def run_execution_result(self, execution_error=None, report=None):
        capture = SimpleNamespace(release=unittest.mock.Mock())
        executor = unittest.mock.Mock()
        executor.execute.side_effect = execution_error
        def solve(*args):
            (self.output/'plan.json').write_text('{"ready_for_motion":true}',encoding='utf8')
            return self.observation(0)
        config = {'execution_backend':'dual_grbl_stm32','auto_max_retries':1,
                  'auto_fallback_enabled':False,'final_verification_enabled':True}
        with patch.object(vision,'load_calibration',return_value=np.eye(3)), \
             patch.object(vision,'open_camera',return_value=capture), \
             patch.object(vision,'capture_immediate_frame',return_value=(self.frames[0],0)), \
             patch.object(vision,'_stable_auto_detect_and_solve',side_effect=solve), \
             patch.object(vision,'DualSerialExecutor',return_value=executor), \
             patch.object(vision,'capture_post_motion_frame',return_value=self.frames[0]), \
             patch.object(vision,'verify_final_frame',return_value=report or {'status':'pass'}), \
             patch.object(vision.cv2,'destroyAllWindows'):
            try:
                vision.run_auto(config,None,'fake',False,True)
            finally:
                capture.release.assert_called_once()
                executor.close.assert_called_once()

    def test_execution_failure_cannot_be_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError,'自动执行未完成'):
            self.run_execution_result(vision.DualExecutionError('R axis failed'))

    def test_failed_final_photo_cannot_be_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError,'最终视觉复核失败'):
            self.run_execution_result(report={'status':'fail'})

    def test_completed_execution_and_passing_photo_succeed(self):
        self.run_execution_result()


if __name__ == '__main__':
    unittest.main(verbosity=2)
