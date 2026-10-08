import copy
import json
from pathlib import Path
import tempfile
import unittest
import threading
import time
from unittest.mock import patch

import cv2
import numpy as np

import approved_plan as ap
import main as vision
from dual_serial_executor import DualExecutionError, DualSerialExecutor, _Wire
from test_dual_serial_executor import BASE_CONFIG, FakeFactory, plan


class ApprovedPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = copy.deepcopy(BASE_CONFIG)
        self.config.update(pixels_per_mm=1., paper_width_mm=100., paper_height_mm=100.,
                           camera_corners=[[0,0],[99,0],[99,99],[0,99]])
        self.image = np.full((100,100,3), 200, np.uint8)
        payload = plan()
        payload.update(failure_reasons=["花纹相似度0.40<0.55，正式执行被锁定"])
        for i, p in enumerate(payload["pieces"]):
            p.update(id=i+1, transform_3x3=np.eye(3).tolist(),
                     source_contour_mm=[[10,10],[20,10],[20,20],[10,20]],
                     center_mm=[15,15], area_mm2=100., size_mm=[10,10])
        self.saved = ap.seal_plan(payload, self.image, self.image, self.config, self.root)
        self.path = Path(self.saved["artifact_path"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_plan_is_not_motion_ready(self):
        self.assertFalse(self.saved["ready_for_motion"])
        self.assertTrue(self.saved["pattern_review_required"])
        self.assertEqual(self.saved["approval_state"], "review_required")
        ap.validate_plan(self.path, self.config, self.saved["plan_id"])

    def test_plan_coordinates_replaced(self):
        changed = copy.deepcopy(self.saved)
        changed["pieces"][0]["source_pick_mm"][0] += 1
        ap.atomic_json(self.path, changed)
        with self.assertRaisesRegex(RuntimeError,"内容"):
            ap.validate_plan(self.path, self.config)

    def test_wrong_plan_id(self):
        with self.assertRaisesRegex(RuntimeError,"编号"):
            ap.validate_plan(self.path, self.config, "different")

    def test_config_change_revokes_plan(self):
        changed = copy.deepcopy(self.config)
        changed["xy_transform"]["bias_mm"][0] += 1
        with self.assertRaisesRegex(RuntimeError,"配置"):
            ap.validate_plan(self.path, changed)

    def test_preview_change_revokes_plan(self):
        (self.path.parent / "assembly.png").write_bytes(b"replacement")
        with self.assertRaisesRegex(RuntimeError,"预览"):
            ap.validate_plan(self.path, self.config)

    def test_geometry_cannot_be_manually_approved(self):
        payload = copy.deepcopy(self.saved)
        payload["failure_reasons"].append("P1目标碎片重叠率超限")
        saved = ap.seal_plan(payload,self.image,self.image,self.config,self.root)
        self.assertEqual(saved["approval_state"],"blocked")
        with self.assertRaises(RuntimeError):
            ap.validate_plan(saved["artifact_path"],self.config)

    def test_used_plan_cannot_open_serial_again(self):
        (self.path.parent/"consumed.json").write_text('{}')
        with patch.object(ap,"DualSerialExecutor") as executor:
            with self.assertRaises(FileExistsError):
                ap.run_session(self.path,lambda:self.config,self.root/'config.json',self.root/'session',0,
                               self.saved['plan_id'])
        executor.assert_not_called()

    def test_scene_moved_and_changed_contours_rejected(self):
        class Piece:
            area_mm2=100.;width_mm=10.;height_mm=10.
            contour=np.array([[10,10],[20,10],[20,20],[10,20]],np.int32).reshape(-1,1,2)
            def center_mm(self,ppm):return (15.,15.)
        pieces=[Piece(),Piece(),Piece()]
        self.assertTrue(ap.scene_matches(pieces,self.saved,self.config)[0])
        pieces[0].center_mm=lambda ppm:(19.,15.)
        self.assertFalse(ap.scene_matches(pieces,self.saved,self.config)[0])
        pieces[0].center_mm=lambda ppm:(15.,15.)
        pieces[0].contour=np.array([[10,10],[26,10],[20,20],[10,20]],np.int32).reshape(-1,1,2)
        self.assertFalse(ap.scene_matches(pieces,self.saved,self.config)[0])

    def test_scene_matches_ignores_unstable_bounding_rectangle_size(self):
        class Piece:
            area_mm2 = 100.
            width_mm = 16.
            height_mm = 5.
            contour = np.array([[10,10],[20,10],[20,20],[10,20]], np.int32).reshape(-1,1,2)
            def center_mm(self, ppm):
                return (15., 15.)
        passed, reason = ap.scene_matches([Piece(), Piece(), Piece()], self.saved, self.config)
        self.assertTrue(passed, reason)

    def test_fresh_scene_check_does_not_solve(self):
        with patch.object(vision,"capture_immediate_frame",return_value=(self.image,0)), \
             patch.object(vision,"warp_paper",return_value=self.image), \
             patch.object(vision,"detect_pieces",return_value=([],None)), \
             patch.object(vision,"_auto_detect_and_solve") as solve:
            capture=type('Capture',(),{'read':lambda s:(True,self.image)})()
            with self.assertRaisesRegex(RuntimeError,"卡片"):
                ap.fresh_scene_check(capture,self.saved,self.config,self.root)
        solve.assert_not_called()


class SessionProtocolTests(unittest.TestCase):
    def config(self,accepted=True):
        c=copy.deepcopy(BASE_CONFIG)
        c.update(require_approved_session=True,calibration_complete=accepted,
                 calibration_state=dict.fromkeys(('camera','xy','z','rotation'),True),
                 accepted_machine_bounds_mm=[[0,0],[300,300]])
        c['motion']['origin_mode']='manual'
        c['mechanical_acceptance']={'operator_confirmed':True,'config_sha256':ap.config_digest(c)}
        return c

    def approved(self):
        p=plan();p.update(plan_id='fixed',approval_state='approved');return p

    def test_fragmented_frames_buffer_until_newline(self):
        factory=FakeFactory();wire=_Wire('stm',115200,factory)
        factory.ports['stm'].responses=[b'ACK,',b'PING\r',b'\nACK,STOP\n']
        self.assertEqual(wire.read(.1),'ACK,PING')
        self.assertEqual(wire.read(.1),'ACK,STOP')

    def test_empty_run_without_global_acceptance_never_lowers_or_energizes(self):
        factory=FakeFactory();ex=DualSerialExecutor(self.config(False),factory,empty_run=True)
        ex.open();ex.confirm_origin('fixed',ex.session_id);ex.execute(self.approved());ex.close()
        self.assertFalse(any('G91' in s for s in factory.ports['grbl'].writes))
        self.assertNotIn('MAGNET,1',factory.ports['stm'].writes)
        self.assertNotIn('AUX,1',factory.ports['stm'].writes)
        self.assertEqual(len([s for s in factory.ports['stm'].writes if s.startswith('ROTATE')]),6)

    def test_boolean_alone_cannot_authorize(self):
        factory=FakeFactory();ex=DualSerialExecutor(self.config(),factory)
        ex.open()
        with self.assertRaisesRegex(DualExecutionError,'批准'):ex.execute(self.approved())
        self.assertFalse(any(s.startswith('$J') for s in factory.ports['grbl'].writes));ex.close()

    def test_approval_consumed_once(self):
        factory=FakeFactory();ex=DualSerialExecutor(self.config(),factory)
        ex.open();ex.confirm_origin('fixed',ex.session_id);ex.execute(self.approved())
        with self.assertRaisesRegex(DualExecutionError,'批准'):ex.execute(self.approved())
        ex.close()

    def test_wrong_connection_cannot_confirm(self):
        ex=DualSerialExecutor(self.config(),FakeFactory());ex.open()
        with self.assertRaisesRegex(DualExecutionError,'会话'):ex.confirm_origin('fixed','old')
        ex.close()

    def test_reset_invalidates_origin(self):
        factory=FakeFactory();ex=DualSerialExecutor(self.config(),factory);ex.open()
        ex.confirm_origin('fixed',ex.session_id)
        factory.ports['grbl'].responses=[b'Grbl 1.1h\n']
        from dual_serial_executor import ControllerResetError
        with self.assertRaises(ControllerResetError):ex.grbl_status()
        self.assertFalse(ex.origin_valid);self.assertTrue(ex.locked);ex.close()

    def test_status_interleaved_before_command_ack(self):
        factory=FakeFactory();ex=DualSerialExecutor(self.config(),factory);ex.open()
        factory.ports['grbl'].responses=[b'<Idle|MPos:0,0,0|FS:0,0>\n']
        result=ex._grbl_command('$J=G90 G21 G53 X10 F300')
        self.assertEqual(result['state'],'Idle');ex.close()

    def test_motion_outside_accepted_bounds_rejected(self):
        c=self.config();c['accepted_machine_bounds_mm']=[[0,0],[100,100]]
        factory=FakeFactory();ex=DualSerialExecutor(c,factory);ex.open()
        ex.confirm_origin('fixed',ex.session_id)
        with self.assertRaisesRegex(DualExecutionError,'范围'):ex.execute(self.approved())
        self.assertFalse(any(s.startswith('$J') for s in factory.ports['grbl'].writes));ex.close()

    def test_calibration_boolean_without_evidence_rejected(self):
        c=self.config();c.pop('mechanical_acceptance')
        with self.assertRaisesRegex(DualExecutionError,'验收记录'):DualSerialExecutor(c,FakeFactory())

    def test_direct_run_uses_saved_calibration_without_empty_acceptance(self):
        c=self.config(False);c['require_empty_run']=False
        c.pop('mechanical_acceptance');c.pop('accepted_machine_bounds_mm')
        c['motion']['xy_feed_mm_min']=1200.;c['z']['feed_mm_min']=240.
        factory=FakeFactory();ex=DualSerialExecutor(c,factory)
        ex.open();ex.confirm_origin('fixed',ex.session_id);ex.execute(self.approved());ex.close()
        commands=[s for s in factory.ports['grbl'].writes if s.startswith('$J')]
        self.assertTrue(any('G91' in s for s in commands))
        self.assertTrue(all('F240.0' in s if 'G91' in s else 'F1200.0' in s for s in commands))
        self.assertEqual(factory.ports['stm'].writes.count('MAGNET,1'),3)

    def test_direct_run_still_requires_all_calibration_components(self):
        for component in ('camera','xy','z','rotation'):
            with self.subTest(component=component):
                c=self.config(False);c['require_empty_run']=False
                c['calibration_state'][component]=False
                with self.assertRaisesRegex(DualExecutionError,'标定状态未完成'):
                    DualSerialExecutor(c,FakeFactory())
        c=self.config(False);c['require_empty_run']=False;c.pop('calibration_state')
        with self.assertRaisesRegex(DualExecutionError,'分项标定状态'):
            DualSerialExecutor(c,FakeFactory())

    def test_direct_run_still_requires_current_session_origin(self):
        c=self.config(False);c['require_empty_run']=False
        factory=FakeFactory();ex=DualSerialExecutor(c,factory);ex.open()
        try:
            with self.assertRaisesRegex(DualExecutionError,'批准'):ex.execute(self.approved())
            self.assertFalse(any(s.startswith('$J') for s in factory.ports['grbl'].writes))
        finally:ex.close()


class WholeSessionTests(unittest.TestCase):
    """Exercise real filesystem IPC and controller sequencing with fake hardware."""
    def setUp(self):
        ApprovedPlanTests.setUp(self)
        self.config.update(require_approved_session=True, calibration_complete=False,
                           calibration_state=dict.fromkeys(('camera','xy','z','rotation'),True))
        self.config['motion']['origin_mode']='manual'
        payload=plan();payload['failure_reasons']=[]
        for i,p in enumerate(payload['pieces']):
            p.update(id=i+1,transform_3x3=np.eye(3).tolist(),
                     source_contour_mm=[[10,10],[20,10],[20,20],[10,20]],
                     center_mm=[15,15],area_mm2=100.,size_mm=[10,10])
        saved=ap.seal_plan(payload,self.image,self.image,self.config,self.root)
        self.path=Path(saved['artifact_path']);self.saved=saved
        self.config_path=self.root/'config.json';ap.atomic_json(self.config_path,self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_replaced_valid_plan_is_rejected_before_ports_open(self):
        with patch.object(ap,'DualSerialExecutor') as executor:
            with self.assertRaisesRegex(RuntimeError,'编号'):
                ap.run_session(self.path,lambda:self.config,self.config_path,
                               self.root/'wrong_session',0,'different-reviewed-plan')
            executor.assert_not_called()
        self.assertFalse((self.path.parent/'consumed.json').exists())

    def run_full_session(self,final_pass=True):
        directory=self.root/'session';factory=FakeFactory();done=threading.Event();confirmed=[]
        def operator():
            seen=set()
            while not done.wait(.02):
                try:s=json.loads((directory/'status.json').read_text('utf8'))
                except (OSError,ValueError):continue
                key=(s['stage'],s['sequence'])
                if key not in seen and s['stage'].startswith('await_'):
                    seen.add(key);confirmed.append(s['stage'])
                    ap.atomic_json(directory/'command.json',{k:s[k] for k in ('session_id','plan_id','sequence')}|{'action':'confirm'})
        worker=threading.Thread(target=operator);worker.start()
        cap=type('Cap',(),{'release':lambda s:None})()
        real_executor=DualSerialExecutor
        def executor(config,empty_run=False):
            return real_executor(config,serial_factory=factory,empty_run=empty_run)
        try:
            with patch.object(ap,'DualSerialExecutor',side_effect=executor), \
                 patch.object(vision,'open_camera',return_value=cap), \
                 patch.object(ap,'fresh_scene_check',return_value=np.eye(3)), \
                 patch.object(vision,'capture_post_motion_frame',return_value=self.image), \
                 patch.object(vision,'verify_final_frame',return_value={'status':'pass' if final_pass else 'fail'}):
                result=ap.run_session(self.path,lambda:json.loads(self.config_path.read_text('utf8')),
                                      self.config_path,directory,0,self.saved['plan_id'])
            return result,factory,confirmed
        finally:
            done.set();worker.join(2)

    def test_whole_empty_then_real_keeps_ports_and_records_separate_acceptance(self):
        result,factory,confirmed=self.run_full_session()
        self.assertEqual(result,0)
        self.assertEqual(confirmed,['await_origin','await_empty_accept','await_origin','await_physical_accept'])
        stm=factory.ports['stm'].writes
        self.assertEqual(stm.count('MAGNET,1'),3)
        self.assertEqual(len([s for s in stm if s.startswith('ROTATE,')]),12)
        config=json.loads(self.config_path.read_text('utf8'))
        self.assertTrue(config['calibration_complete'])
        result=json.loads((self.root/'puzzle_acceptance.json').read_text('utf8'))
        self.assertEqual(result['consecutive_passes'],1);self.assertFalse(result['complete'])

    def test_whole_direct_run_has_one_origin_and_no_empty_path(self):
        self.config['require_empty_run']=False
        self.saved['config_sha256']=ap.config_digest(self.config)
        self.saved['content_sha256']=ap.immutable_digest(self.saved)
        ap.atomic_json(self.path,self.saved);ap.atomic_json(self.config_path,self.config)
        result,factory,confirmed=self.run_full_session()
        self.assertEqual(result,0)
        self.assertEqual(confirmed,['await_origin','await_physical_accept'])
        stm=factory.ports['stm'].writes
        self.assertEqual(stm.count('MAGNET,1'),3)
        self.assertEqual(len([s for s in stm if s.startswith('ROTATE,')]),6)
        self.assertFalse((self.root/'session/empty_run.json').exists())
        self.assertFalse((self.root/'session/approved_empty.json').exists())
        self.assertTrue((self.root/'session/approved_real.json').exists())
        self.assertEqual(json.loads(self.config_path.read_text('utf8')),self.config)
        status=json.loads((self.root/'session/status.json').read_text('utf8'))
        self.assertEqual(status['stage'],'complete');self.assertFalse(status['empty_run'])

    def test_failed_final_photo_never_reports_complete(self):
        with self.assertRaisesRegex(RuntimeError,'复拍未通过'):self.run_full_session(False)
        status=json.loads((self.root/'session/status.json').read_text('utf8'))
        self.assertEqual(status['stage'],'failed')
        history=json.loads((self.root/'puzzle_acceptance.json').read_text('utf8'))
        self.assertFalse(history['complete'])


class FinalPhotoMaskTests(unittest.TestCase):
    def test_one_mm_seams_stay_separate_and_touching_is_not_invented(self):
        config=vision.load_config()
        config.update(puzzle_mode='generic_geometry',pixels_per_mm=4.,source_region='top',
                      morphology_kernel_mm=.3,generic_contour_close_mm=.3,
                      paper_width_mm=210.,paper_height_mm=297.,separator_line_y_mm=148.5)
        for gap,expected in [(4,3),(0,1)]:
            mask=np.zeros((1188,840),np.uint8)
            for i in range(3):
                x=100+i*(80+gap);mask[100:220,x:x+80]=255
            cleaned=vision.clean_piece_mask(mask,config)
            refined=vision.refine_generic_piece_mask(cleaned,config)
            self.assertEqual(vision.count_piece_candidates(refined,config),expected)


class CameraProfileTests(unittest.TestCase):
    def test_explicit_resolution_is_set_before_first_read(self):
        calls=[]
        class Capture:
            def set(self,key,value):calls.append((key,value));return True
            def read(self):
                self_ready=any(k==cv2.CAP_PROP_FRAME_WIDTH and v==1280 for k,v in calls)
                if not self_ready:raise AssertionError('read before profile')
                return True,np.zeros((720,1280,3),np.uint8)
        config={'camera_profile_required':True,'camera_width':1280,'camera_height':720,
                'camera_calibration_image_size':[1280,720]}
        frame=vision._probe_opened_capture(config,Capture())
        self.assertEqual(frame.shape,(720,1280,3))

    def test_resolution_fallback_cannot_use_wrong_corners(self):
        class Capture:
            def set(self,key,value):return True
            def read(self):return True,np.zeros((480,640,3),np.uint8)
        with self.assertRaisesRegex(RuntimeError,'分辨率'):
            vision._probe_opened_capture({'camera_profile_required':True,'camera_width':1280,
                                         'camera_height':720},Capture())


if __name__=='__main__':unittest.main()
