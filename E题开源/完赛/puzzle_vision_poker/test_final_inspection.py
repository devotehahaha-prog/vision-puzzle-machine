"""Exercise final-photo acceptance on real synthetic pixels, without hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import main as vision


class FinalInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.cfg = {
            'paper_width_mm': 210., 'paper_height_mm': 297., 'pixels_per_mm': 2.,
            'segmentation_mode': 'white_piece', 'puzzle_mode': 'generic_geometry',
            'white_piece_threshold_mode': 'fixed', 'white_piece_value_min': 200,
            'white_piece_saturation_max': 55, 'morphology_kernel_mm': .3,
            'source_region': 'bottom', 'target_region': 'top',
            'target_region_top_mm': 148.5, 'separator_line_half_width_mm': 3.,
            'min_piece_area_mm2': 100., 'max_piece_area_mm2': 8000.,
            'min_solidity': .3, 'max_pieces': 4,
            'rotation_match_pixels_per_mm': 2., 'rotation_search_step_deg': 5.,
            'rotation_refine_step_deg': 1., 'final_verify_max_position_error_mm': 3.,
            'final_verify_max_rotation_error_deg': 3.,
        }
        self.polygons = [np.array([[x,y],[x+25,y],[x+25,y+20],[x,y+20]],float)
                         for x,y in ((20,20),(65,20),(20,65),(65,65))]

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, polygons):
        image = np.zeros((594,420,3),np.uint8)
        image[:]=(50,110,50)
        for p in polygons:
            cv2.fillPoly(image,[np.rint(p*2).astype(np.int32)],(245,245,245))
        return image

    def run_photo(self, actual=None, targets=None, config=None):
        targets=self.polygons if targets is None else targets
        actual=targets if actual is None else actual
        plan={'piece_count':len(targets),'pieces':[
            {'id':i+1,'template_id':f'G{i+1}','target_polygon_mm':p.tolist()}
            for i,p in enumerate(targets)]}
        path=self.output/'plan.json'
        path.write_text(json.dumps(plan),encoding='utf8')
        out=io.StringIO()
        with patch.object(vision,'OUTPUT_DIR',self.output),contextlib.redirect_stdout(out):
            report=vision.verify_final_frame(self.frame(actual),np.eye(3),config or self.cfg,path)
        return report,out.getvalue()

    def test_four_separate_correct_pieces_pass(self):
        report,_=self.run_photo()
        self.assertEqual(report['status'],'pass')
        self.assertEqual(report['detected_piece_count'],4)
        self.assertEqual(report['source_leftover_count'],0)
        self.assertTrue(all(not m['measurement_tentative'] for m in report['matches']))

    def test_fifth_piece_cannot_be_hidden_by_planning_cap(self):
        extra=np.array([[120,30],[140,30],[140,45],[120,45]],float)
        actual=self.polygons+[extra]
        # Reproduce the old bug: the planning detector cap returns only four.
        pieces,_=vision.detect_pieces(self.frame(actual),dict(self.cfg,source_region='top'))
        self.assertEqual(len(pieces),4)
        report,_=self.run_photo(actual)
        self.assertEqual(report['status'],'fail')
        self.assertEqual(report['detected_piece_count'],5)
        self.assertEqual(report['unexpected_piece_count'],1)

    def test_two_extra_pieces_are_counted_exactly(self):
        extras=[p+np.array([110,0]) for p in self.polygons[:2]]
        report,_=self.run_photo(self.polygons+extras)
        self.assertEqual(report['detected_piece_count'],6)
        self.assertEqual(report['unexpected_piece_count'],2)
        self.assertEqual(report['status'],'fail')

    def test_merged_contours_do_not_claim_proven_missing_piece(self):
        actual=list(self.polygons)
        actual[1]=actual[1]+np.array([-20,0])  # touches the first rectangle
        report,output=self.run_photo(actual)
        self.assertEqual(report['status'],'fail')
        self.assertEqual(report['detected_piece_count'],3)
        self.assertTrue(report['unresolved_target_piece_ids'])
        self.assertTrue(all(m['measurement_tentative'] for m in report['matches']))
        self.assertNotIn('漏吸或漏放目标',output)
        self.assertIn('未能独立识别',output)
        self.assertIn('勿用于标定',output)

    def test_empty_scene_fails_without_fabricating_matches(self):
        report,_=self.run_photo([])
        self.assertEqual(report['status'],'fail')
        self.assertEqual(report['unresolved_target_piece_ids'],[1,2,3,4])
        self.assertEqual(report['matches'],[])

    def test_source_leftover_rejects_otherwise_correct_layout(self):
        report,_=self.run_photo(self.polygons+[self.polygons[0]+np.array([0,170])])
        self.assertEqual(report['detected_piece_count'],4)
        self.assertEqual(report['source_leftover_count'],1)
        self.assertEqual(report['status'],'fail')

    def test_reversed_regions_inspect_source_instead_of_target_twice(self):
        targets=[p+np.array([0,155]) for p in self.polygons]
        cfg=dict(self.cfg,source_region='top',target_region='bottom')
        report,_=self.run_photo(targets=targets,config=cfg)
        self.assertEqual(report['status'],'pass')
        self.assertEqual(report['source_leftover_count'],0)
        report,_=self.run_photo(targets+[self.polygons[0]],targets,cfg)
        self.assertEqual(report['status'],'fail')
        self.assertEqual(report['source_leftover_count'],1)

    def test_placement_tolerance_is_not_relaxed(self):
        actual=list(self.polygons)
        actual[0]=actual[0]+np.array([5,0])
        report,_=self.run_photo(actual)
        self.assertEqual(report['status'],'fail')
        moved=next(m for m in report['matches'] if m['target_piece_id']==1)
        self.assertGreater(moved['position_error_mm'],3.)
        self.assertFalse(moved['passed'])

    def test_overlay_labels_ambiguity_without_missing_claim(self):
        actual=list(self.polygons)
        actual[1]=actual[1]+np.array([-20,0])
        with patch.object(vision.cv2,'putText',wraps=cv2.putText) as draw:
            self.run_photo(actual)
        labels=[call.args[1] for call in draw.call_args_list]
        self.assertTrue(any('UNRESOLVED' in label for label in labels))
        self.assertFalse(any('MISSING' in label for label in labels))


if __name__=='__main__':
    unittest.main()
