"""Four-piece live profile: bound plans and simulated controllers only."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import approved_plan as ap
from dual_serial_executor import DualExecutionError, DualSerialExecutor
from test_dual_serial_executor import BASE_CONFIG, FakeFactory, FakePort, plan


def config4():
    c=copy.deepcopy(BASE_CONFIG)
    c.update(puzzle_profile='ordinary',expected_piece_count=4,require_approved_session=True,
             require_empty_run=False,pixels_per_mm=1.,paper_width_mm=100.,paper_height_mm=100.,
             camera_corners=[[0,0],[99,0],[99,99],[0,99]],
             calibration_state=dict.fromkeys(('camera','xy','z','rotation'),True))
    c['motion']['origin_mode']='manual'
    return c


def plan4():
    p=plan()
    fourth=copy.deepcopy(p['pieces'][0]);fourth['move_order']=4
    p['pieces'].append(fourth)
    p.update(puzzle_profile='ordinary',piece_count=4,plan_id='ordinary',approval_state='approved',failure_reasons=[])
    for i,piece in enumerate(p['pieces']):
        piece.update(id=i+1,transform_3x3=np.eye(3).tolist(),
                     source_contour_mm=[[10,10],[20,10],[20,20],[10,20]],
                     center_mm=[15,15],area_mm2=100.,size_mm=[10,10])
    return p


class OrdinaryExecutionTests(unittest.TestCase):
    def test_poker_profile_cannot_be_changed_to_four_by_count_alone(self):
        c=copy.deepcopy(BASE_CONFIG);c['expected_piece_count']=4
        with self.assertRaises(DualExecutionError):DualSerialExecutor(c,FakeFactory())

    def test_four_piece_plan_requires_matching_profile_and_session(self):
        factory=FakeFactory();ex=DualSerialExecutor(config4(),factory,sleeper=lambda _:None)
        ex.open()
        try:
            with self.assertRaisesRegex(DualExecutionError,'批准'):ex.execute(plan4())
            ex.confirm_origin('ordinary',ex.session_id)
            p=plan4();p['puzzle_profile']='poker'
            with self.assertRaisesRegex(DualExecutionError,'当前模式'):ex.execute(p)
            self.assertFalse(any(s.startswith('$J') for s in factory.ports['grbl'].writes))
        finally:ex.close()

    def test_four_piece_sequence_and_single_use_authorization(self):
        factory=FakeFactory();ex=DualSerialExecutor(config4(),factory,sleeper=lambda _:None)
        ex.open()
        try:
            ex.confirm_origin('ordinary',ex.session_id);ex.execute(plan4())
            stm=factory.ports['stm'].writes
            self.assertEqual(stm.count('MAGNET,1'),4)
            self.assertEqual(stm.count('MAGNET,0'),4)
            self.assertEqual(len([x for x in stm if x.startswith('ROTATE,')]),8)
            self.assertEqual(factory.ports['grbl'].xyz,[0.,0.,0.])
            self.assertEqual(factory.ports['stm'].magnet,0)
            with self.assertRaisesRegex(DualExecutionError,'批准'):ex.execute(plan4())
        finally:ex.close()

    def test_missing_fourth_piece_is_rejected_before_motion(self):
        factory=FakeFactory();ex=DualSerialExecutor(config4(),factory,sleeper=lambda _:None)
        ex.open()
        try:
            ex.confirm_origin('ordinary',ex.session_id)
            p=plan4();p['pieces'].pop()
            with self.assertRaisesRegex(DualExecutionError,'4块'):ex.execute(p)
            self.assertFalse(any(s.startswith('$J') for s in factory.ports['grbl'].writes))
        finally:ex.close()

    def test_release_disconnect_stops_and_never_replays_next_piece(self):
        factory=FakeFactory();ex=DualSerialExecutor(config4(),factory,sleeper=lambda _:None)
        ex.open();ex.confirm_origin('ordinary',ex.session_id)
        original=FakePort.write
        def write(port,payload):
            if port.port=='stm' and payload==b'MAGNET,0\n':
                raise OSError('release disconnected')
            return original(port,payload)
        try:
            with patch.object(FakePort,'write',write):
                with self.assertRaises(DualExecutionError):ex.execute(plan4())
            self.assertTrue(ex.locked)
            self.assertEqual(factory.ports['stm'].writes.count('MAGNET,1'),1)
            self.assertIn(b'!\x85',factory.ports['grbl'].raw_writes)
            self.assertIn('STOP',factory.ports['stm'].writes)
        finally:ex.close()

    def test_four_piece_seal_validation_and_missing_piece_scene(self):
        c=config4();image=np.full((100,100,3),200,np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            sealed=ap.seal_plan(plan4(),image,image,c,Path(folder))
            self.assertFalse(sealed['ready_for_motion'])
            self.assertEqual(sealed['approval_state'],'review_required')
            self.assertEqual(len(ap.validate_plan(sealed['artifact_path'],c)['pieces']),4)
            ok,reason=ap.scene_matches([],sealed,c)
            self.assertFalse(ok);self.assertIn('要求4片',reason)
            changed=copy.deepcopy(c);changed['ordinary_entry_sha256']='code changed'
            with self.assertRaisesRegex(RuntimeError,'配置'):ap.validate_plan(sealed['artifact_path'],changed)


if __name__=='__main__':unittest.main()
