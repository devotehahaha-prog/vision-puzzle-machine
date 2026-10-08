"""Regression tests for measured camera-plan failures, without real devices."""
import json
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import main as vision
import generic_solver
import run_real


class PlanGenerationDiagnosticsTests(unittest.TestCase):
    def check_consensus_retry(self, *, reject_retry=False, moved_source=False):
        frames = [np.full((20,20,3), i, np.uint8) for i in range(5)]
        def observation(index, target, moved=False):
            pieces = [SimpleNamespace(center_mm=lambda ppm,j=j:(10.*j+(10. if moved else 0.), 10.),
                                      area_mm2=50., width_mm=5., height_mm=10.) for j in range(3)]
            paths = [SimpleNamespace(piece=p, match=SimpleNamespace(
                target_pick_mm=(10.*j+target, 40.), rotation_deg=0.)) for j,p in enumerate(pieces)]
            return (np.eye(3), frames[index], frames[index][:,:,0], pieces, paths)
        initial = [observation(i, float(i)) for i in range(3)]
        initial.append(observation(3, 20., moved_source))
        retried = observation(3, 1.)
        def detect(frame, config, matrix):
            i=int(frame[0,0,0])
            if '_planning_reference_groups' in config:
                self.assertEqual(i,3)
                self.assertEqual(len(config['_planning_reference_groups']),3)
                return retried
            if i==4:
                raise ValueError('unusable fifth frame')
            return initial[i]
        config={'pixels_per_mm':1., 'expected_piece_count':3, 'planning_polygon_alternatives':True}
        iterator=iter(frames[1:])
        capture=SimpleNamespace(read=lambda:(True,next(iterator)))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(vision,'OUTPUT_DIR',Path(directory)), \
             patch.object(vision,'_auto_detect_and_solve',side_effect=detect) as solve, \
             patch.object(vision,'motion_safety_reasons',side_effect=lambda paths,c:
                          ['unsafe'] if reject_retry and paths is retried[4] else []), \
             patch.object(vision,'draw_preview',return_value=frames[0]), \
             patch.object(vision,'save_outputs') as save, \
             patch.object(vision,'save_failed_detection'):
            if reject_retry or moved_source:
                with self.assertRaisesRegex(RuntimeError,'稳定阈值'):
                    vision._stable_auto_detect_and_solve(capture,frames[0],config,None)
                save.assert_not_called()
            else:
                vision._stable_auto_detect_and_solve(capture,frames[0],config,None)
                save.assert_called_once()
            self.assertEqual(solve.call_count, 5 if moved_source else 6)
            report=json.loads((Path(directory)/'stability_report.json').read_text(encoding='utf8'))
            self.assertEqual(len(report['frame_reports']),5)
            self.assertEqual(report['passed'],not (reject_retry or moved_source))

    def test_consensus_requires_an_independent_fourth_frame(self):
        self.check_consensus_retry()

    def test_consensus_cannot_override_retried_geometry_rejection(self):
        self.check_consensus_retry(reject_retry=True)

    def test_consensus_does_not_retry_physically_moved_piece(self):
        self.check_consensus_retry(moved_source=True)

    def alternatives_fixture(self):
        rows = json.loads(Path(__file__).with_name('plan_alternatives_fixture.json').read_text())
        return [vision.Piece(**dict(row, contour=np.asarray(row['contour'], np.int32))) for row in rows]

    def test_faithful_alternative_recovers_rejected_real_contour(self):
        pieces = self.alternatives_fixture()
        config = run_real.load_production_config()
        with self.assertRaises(ValueError):
            vision.solve_generic_puzzle(pieces, dict(config, planning_polygon_alternatives=False))
        matches = vision.solve_generic_puzzle(pieces, dict(config, planning_polygon_alternatives=True))
        self.assertEqual(len(matches), 3)
        for match in matches:
            self.assertLessEqual(match.fill_error_ratio, config['generic_motion_max_fill_error_ratio'])
            self.assertLessEqual(match.boundary_gap_ratio, config['generic_motion_max_boundary_gap_ratio'])
            self.assertFalse(match.relaxed_override)

    def test_every_alternative_respects_original_area_limit(self):
        config = run_real.load_production_config()
        config['planning_polygon_alternatives'] = True
        for piece in self.alternatives_fixture():
            area = cv2.contourArea(piece.contour) / config['pixels_per_mm'] ** 2
            models = vision._piece_polygon_models_mm(piece, config)
            self.assertLessEqual(len(models), 3)
            for polygon in models:
                self.assertLessEqual(abs(generic_solver.polygon_area(polygon)-area)/area,
                                     config['generic_max_polygon_area_error_ratio'])

    def test_exhausted_alternative_search_does_not_return_unsafe_plan(self):
        config = run_real.load_production_config()
        config.update(planning_polygon_alternatives=True, generic_max_search_states=1)
        with self.assertRaises(ValueError):
            vision.solve_generic_puzzle(self.alternatives_fixture(), config)

    def test_loose_candidate_does_not_prevent_valid_split_search(self):
        fixture = json.loads(Path(__file__).with_name('plan_generation_fixture.json').read_text())
        config = run_real.load_production_config()
        config['generic_min_edge_length_mm'] = 7.
        original = dict(config)
        solution = generic_solver.solve_generic_rectangle(
            [np.asarray(p) for p in fixture['polygons_mm']], config)
        self.assertLessEqual(solution.fill_error_ratio, config['generic_motion_max_fill_error_ratio'])
        self.assertLessEqual(solution.boundary_gap_ratio, config['generic_motion_max_boundary_gap_ratio'])
        self.assertFalse(solution.relaxed_override)
        self.assertEqual(config, original)

    def test_rounded_corner_does_not_gain_unstable_fifth_edge(self):
        fixture = json.loads(Path(__file__).with_name('plan_generation_fixture.json').read_text())
        contour = np.asarray(fixture['rounded_contour_px'], np.float32)
        piece = SimpleNamespace(piece_id=1, contour=contour)
        config = run_real.load_production_config()
        config['planning_polygon_area_tie_tolerance'] = .01
        polygon = vision.piece_polygon_mm(piece, config)
        self.assertEqual(len(polygon), 4)
        actual_area = cv2.contourArea(contour) / config['pixels_per_mm'] ** 2
        self.assertLessEqual(abs(generic_solver.polygon_area(polygon)-actual_area)/actual_area,
                             config['generic_max_polygon_area_error_ratio'])

    def test_four_consistent_frames_are_not_split_by_first_outlier(self):
        values = [-3., 0., 1., 2., 3.]
        for order in itertools.permutations(values):
            group = vision._largest_consistent_group(list(order), lambda a, b: abs(a-b) <= 3.)
            self.assertEqual(set(group), {0., 1., 2., 3.})

    def test_consistency_is_not_transitive_chaining(self):
        group = vision._largest_consistent_group([0., 2., 4., 6., 8.], lambda a, b: abs(a-b) <= 3.)
        self.assertEqual(len(group), 2)

    def test_nearly_equal_x_positions_do_not_swap_piece_identity(self):
        a = [(10., 50., 100., 10., 20.), (10.1, 90., 120., 12., 22.)]
        b = [(9.9, 90., 120., 12., 22.), (10.2, 50., 100., 10., 20.)]
        self.assertTrue(vision._signatures_stable(a, b, {}))
        b[0] = (9.9, 100., 120., 12., 22.)
        self.assertFalse(vision._signatures_stable(a, b, {}))

    def test_triangle_rectangle_switch_is_not_real_shape_change(self):
        contour = np.float32([[[10, 20]], [[40, 30]], [[15, 80]]])
        def piece(w, h, actual=contour):
            return SimpleNamespace(contour=actual, center_mm=lambda ppm:(10., 20.),
                                   area_mm2=100., width_mm=w, height_mm=h)
        a = vision._piece_stability_signature([piece(33., 67.)], 1.)
        b = vision._piece_stability_signature([piece(39., 58.)], 1.)
        self.assertTrue(vision._signatures_stable(a, b, {}))
        changed = contour.copy(); changed[:, 0, 0] *= 1.5
        other = vision._piece_stability_signature([piece(33., 67., changed)], 1.)
        self.assertFalse(vision._signatures_stable(a, other, {}))

    def test_missing_piece_stops_before_geometric_solver(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        config = {'strict_production':True, 'expected_piece_count':3, 'puzzle_mode':'generic_geometry'}
        with patch.object(vision, 'detect_pieces', return_value=([object(), object()], frame[:,:,0])), \
             patch.object(vision, 'number_by_move_order', side_effect=lambda p,c:p), \
             patch.object(vision, 'calibration_quality_reasons', return_value=[]), \
             patch.object(vision, 'piece_border_contacts', return_value=[]), \
             patch.object(vision, 'solve_puzzle') as solve:
            with self.assertRaisesRegex(RuntimeError, '识别到2/3片'):
                vision.process_paper(frame, frame, config, False, save=False)
        solve.assert_not_called()

    def test_all_rejected_frames_refresh_diagnostics_and_invalidate_plan(self):
        frame = np.full((100,100,3), 90, np.uint8)
        mask = np.zeros((100,100), np.uint8)
        config = {'pixels_per_mm':1., 'expected_piece_count':3, 'min_piece_area_mm2':10.}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(vision, 'OUTPUT_DIR', Path(directory)), \
             patch.object(vision, 'resolve_calibration_matrix', return_value=(np.eye(3),'saved')), \
             patch.object(vision, 'warp_paper', return_value=frame), \
             patch.object(vision, 'mask_in_paper', return_value=(mask,mask)), \
             patch.object(vision, 'process_paper', side_effect=RuntimeError('识别到2/3片')):
            root = Path(directory)
            (root/'plan.json').write_text('{"ready_for_motion":true}')
            capture = SimpleNamespace(read=lambda:(True,frame))
            with self.assertRaisesRegex(RuntimeError, '仅0帧'):
                vision._stable_auto_detect_and_solve(capture, frame, config, None)
            plan = json.loads((root/'plan.json').read_text(encoding='utf8'))
            self.assertFalse(plan['ready_for_motion'])
            self.assertEqual(plan['pieces'], [])
            detail = json.loads((root/'detection_failure.json').read_text(encoding='utf8'))
            self.assertEqual(detail['reason'], '识别到2/3片')
            self.assertTrue((root/'detection_preview.jpg').is_file())
            saved = cv2.imdecode(np.fromfile(root/'original.jpg', np.uint8), 1)
            self.assertLess(abs(float(saved.mean())-90), 1)

    def test_denoising_does_not_mutate_camera_frame(self):
        frame = np.arange(30000, dtype=np.uint8).reshape(100,100,3)
        original = frame.copy()
        cfg = {'segmentation_mode':'background_inverse', 'raw_background_denoise_sigma_px':1.}
        with patch.object(vision, 'warp_paper', side_effect=lambda f,m,c:f), \
             patch.object(vision, 'make_piece_mask', return_value=np.zeros((100,100),np.uint8)):
            vision.mask_in_paper(frame, np.eye(3), cfg)
        np.testing.assert_array_equal(frame, original)

    def test_stricter_outline_failure_retains_original_short_edge_model(self):
        frame=np.zeros((100,100,3),np.uint8)
        config={'strict_production':True, 'expected_piece_count':3, 'puzzle_mode':'generic_geometry',
                'generic_min_edge_length_mm':4., 'planning_polygon_min_edge_candidates_mm':[7.]}
        match=SimpleNamespace()
        with patch.object(vision,'detect_pieces',return_value=([1,2,3],frame[:,:,0])), \
             patch.object(vision,'number_by_move_order',side_effect=lambda p,c:p), \
             patch.object(vision,'calibration_quality_reasons',return_value=[]), \
             patch.object(vision,'piece_border_contacts',return_value=[]), \
             patch.object(vision,'solve_puzzle',side_effect=[ValueError('real short edge'),[match]]) as solve, \
             patch.object(vision,'plan_all_paths',return_value=[object()]), \
             patch.object(vision,'motion_safety_reasons',return_value=[]):
            vision.process_paper(frame,frame,config,False,save=False)
        self.assertEqual([call.args[1]['generic_min_edge_length_mm'] for call in solve.call_args_list],[7.,4.])
        self.assertEqual(config['generic_min_edge_length_mm'],4.)
        self.assertEqual(match.polygon_min_edge_mm,4.)

    def test_alternate_outline_cannot_bypass_geometry_rejection(self):
        frame=np.zeros((100,100,3),np.uint8)
        config={'strict_production':True, 'expected_piece_count':3, 'puzzle_mode':'generic_geometry',
                'generic_min_edge_length_mm':4., 'planning_polygon_min_edge_candidates_mm':[7.]}
        with patch.object(vision,'detect_pieces',return_value=([1,2,3],frame[:,:,0])), \
             patch.object(vision,'number_by_move_order',side_effect=lambda p,c:p), \
             patch.object(vision,'calibration_quality_reasons',return_value=[]), \
             patch.object(vision,'piece_border_contacts',return_value=[]), \
             patch.object(vision,'solve_puzzle',return_value=[SimpleNamespace()]), \
             patch.object(vision,'plan_all_paths',return_value=[object()]), \
             patch.object(vision,'motion_safety_reasons',return_value=['overlap']) as gate:
            with self.assertRaisesRegex(RuntimeError,'overlap'):
                vision.process_paper(frame,frame,config,False,save=False)
        self.assertEqual(gate.call_count,2)


if __name__ == '__main__':
    unittest.main()
