import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import calibration_preview
import layout
import start_page
import camera_job


class CalibrationPreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120,80,3), 90, np.uint8)
        self.config = {'paper_width_mm':80., 'paper_height_mm':120., 'pixels_per_mm':1.,
                       'camera_calibration_image_size':[80,120], 'expected_piece_count':3,
                       'camera_calibration':{'perspective_matrix':np.eye(3).tolist(), 'saved_at':'test'}}
        self.vision = SimpleNamespace(
            load_calibration=lambda c:np.eye(3),
            resolve_calibration_matrix=lambda f,c,m:(m,'saved'),
            paper_size_px=lambda c:(80,120),
            warp_paper=lambda f,m,c:cv2.warpPerspective(f,m,(80,120)),
            mask_in_paper=lambda f,m,c:(None,np.zeros((120,80),np.uint8)),
            detect_pieces=lambda p,c,mask_override:([],mask_override))

    def test_views_use_effective_matrix_without_editing_frame_or_config(self):
        original = self.frame.copy()
        before = json.dumps(self.config, sort_keys=True)
        for view in ('raw','grid','mask'):
            display, info = calibration_preview.render_calibration_preview(self.frame,self.vision,self.config,view)
            self.assertEqual(display.shape, self.frame.shape)
            self.assertEqual(info['calibration_source'],'production_config')
            np.testing.assert_allclose(info['image_corners'],[[0,0],[79,0],[79,119],[0,119]])
        np.testing.assert_array_equal(self.frame, original)
        self.assertEqual(json.dumps(self.config,sort_keys=True),before)

    def test_mismatched_resolution_is_not_silently_rescaled(self):
        self.config['camera_calibration_image_size']=[1280,720]
        with self.assertRaisesRegex(ValueError,'分辨率不匹配'):
            calibration_preview.render_calibration_preview(self.frame,self.vision,self.config)

    def test_invalid_matrix_is_rejected(self):
        self.vision.load_calibration=lambda c:np.zeros((3,3))
        with self.assertRaisesRegex(ValueError,'矩阵无效'):
            calibration_preview.render_calibration_preview(self.frame,self.vision,self.config)

    def test_preview_cannot_save_calibration_or_trigger_motion(self):
        state=layout.StartPageState(default_mode=layout.MODE_POKER)
        state.open_calibration_preview()
        self.assertIsNone(state.on_points([(300,200)]))
        state.on_points([])
        self.assertIsNone(state.on_points([(230,448)]))
        for view,rect,label in layout.CALIB_VIEW_BUTTONS:
            state.on_points([])
            self.assertEqual(state.on_points([(rect[0]+5,rect[1]+5)]),('calibration_view',view))

    def test_preview_entry_and_busy_lockout(self):
        state=layout.StartPageState(default_mode=layout.MODE_POKER)
        x,y,_,_=layout.CALIB_PREVIEW_BTN
        self.assertEqual(state.on_points([(x+5,y+5)]),'calib_preview')
        state.on_points([]);state.busy=True
        self.assertIsNone(state.on_points([(x+5,y+5)]))
        self.assertFalse(state.open_calibration_preview())

    def test_home_tool_buttons_do_not_overlap(self):
        rects=[layout.CALIB_BTN,layout.CALIB_PREVIEW_BTN,layout.TUNE_BTN,layout.LIVE_BTN]
        for a,b in zip(rects,rects[1:]):self.assertLessEqual(a[0]+a[2],b[0])
        self.assertLessEqual(rects[-1][0]+rects[-1][2],800)

    def test_old_diagnostics_are_not_shown_as_current_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'output';out.mkdir()
            (out/'detection_failure.json').write_text(json.dumps({'time':10.,'reason':'old'}))
            (out/'detection_preview.jpg').write_bytes(b'old image')
            (out/'plan.json').write_text(json.dumps({'generated_at_epoch':30.,'failure_reasons':['current']}))
            message,image=start_page.read_planning_failure(root,20.)
            self.assertEqual(message,'current');self.assertIsNone(image)

    def test_current_failure_message_and_image_are_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'output';out.mkdir()
            (out/'detection_failure.json').write_text(json.dumps({'time':30.,'reason':'two pieces'}))
            (out/'detection_preview.jpg').write_bytes(b'image')
            message,image=start_page.read_planning_failure(root,20.)
            self.assertEqual(message,'two pieces');self.assertEqual(image,out/'detection_preview.jpg')

    def test_camera_error_removes_previous_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'spi_preview.jpg').write_bytes(b'previous frame')
            (root/'stop').touch()
            fake=SimpleNamespace(load_config=lambda:{}, open_camera=Mock(side_effect=RuntimeError('camera unavailable')))
            args=SimpleNamespace(workdir=root,project=root,vision='poker',mode='calib_preview')
            with patch.object(camera_job,'load_vision',return_value=fake):
                camera_job.cmd_real_preview(args)
            self.assertFalse((root/'spi_preview.jpg').exists())
            self.assertFalse(json.loads((root/'status.json').read_text())['ok'])
