"""Numerical and rejection regressions for the solver's reusable calculations."""
import math
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import generic_solver as solver


def reference_boundary(polygons, width, height, tolerance):
    errors=[];all_ok=True
    for polygon in polygons:
        best=math.inf;ok=False
        for edge in solver.polygon_edges(polygon):
            for axis,side in ((0,0.),(0,width),(1,0.),(1,height)):
                d=np.abs(edge[:,axis]-side)
                best=min(best,float(np.mean(d)))
                ok=ok or float(np.max(d))<=tolerance
        errors.append(best if math.isfinite(best) else math.hypot(width,height))
        all_ok=all_ok and ok
    return all_ok,float(np.mean(errors)) if errors else 0.


def reference_overlap(polygons,ppm):
    if len(polygons)<=1:return 0.
    ppm=max(1.,float(ppm));minimum=np.min(np.vstack(polygons),axis=0)-2.
    shifted=[(p-minimum)*ppm for p in polygons];maximum=np.max(np.vstack(shifted),axis=0)
    shape=(max(3,math.ceil(float(maximum[1]))+3),max(3,math.ceil(float(maximum[0]))+3))
    coverage=np.zeros(shape,np.uint8);union=np.zeros_like(coverage)
    for p in shifted:
        layer=np.zeros(shape,np.uint8);cv2.fillPoly(layer,[np.rint(p).astype(np.int32)],1)
        union=np.maximum(union,layer)
        coverage=np.minimum(255,coverage+cv2.erode(layer,np.ones((3,3),np.uint8))).astype(np.uint8)
    return float(np.count_nonzero(coverage>1)/np.count_nonzero(union)) if np.any(union) else 1.


class SolverSpeedTests(unittest.TestCase):
    def test_vector_boundary_matches_original_on_random_and_closed_contours(self):
        rng=np.random.default_rng(1938)
        for _ in range(80):
            ps=[rng.uniform(-10,120,(n,2)) for n in (3,4,5)]
            ps[0]=np.vstack((ps[0],ps[0][0]))
            for t in (0.,.3,4.,100.):
                expected=reference_boundary(ps,100.,60.,t)
                actual=solver._outside_edge_quality(ps,100.,60.,t)
                self.assertEqual(actual[0],expected[0]);self.assertAlmostEqual(actual[1],expected[1],places=12)

    def test_boundary_limit_is_inclusive_and_not_relaxed(self):
        rectangle=np.array([[0,3],[10,3],[10,13],[0,13]],float)+[20,0]
        self.assertTrue(solver._outside_edge_quality([rectangle],100,60,3)[0])
        self.assertFalse(solver._outside_edge_quality([rectangle],100,60,2.999)[0])

    def test_overlap_pixels_match_original_for_touching_and_overlapping_shapes(self):
        p=np.array([[0,0],[20,0],[20,10],[0,10]],float)
        for dx in (0.,.1,5.,19.9,20.,20.1,40.):
            for ppm in (1.,2.,3.):
                polygons=[p,p+[dx,0],p+[0,15],p+[25,20]]
                self.assertEqual(solver._interior_overlap_ratio(polygons,ppm),reference_overlap(polygons,ppm))

    def test_validated_pose_key_retains_old_quantization(self):
        for angle in (-math.pi,-.0001,0.,.2,math.pi/2,math.pi):
            poses={0:np.eye(3),1:solver.rigid_matrix(angle,[50.25,-20.25])}
            args=(poses,frozenset({(0,2),(1,1)}),.5,.5)
            self.assertEqual(solver._pose_state_key(*args),solver._pose_state_key(*args,validated_rigid=True))

    def test_external_pose_key_still_rejects_nonrigid_matrices(self):
        with self.assertRaises(ValueError):
            solver._pose_state_key({0:np.diag([2.,1.,1.])},frozenset(),.5,.5)

    def test_edge_alignment_results_are_not_shared_between_scenes(self):
        config={'generic_enable_closure_refine':False,'generic_enable_collinear_split':False,
                'generic_search_timeout_seconds':8.,'generic_target_long_min_mm':90.,
                'generic_target_long_max_mm':120.,'generic_target_short_min_mm':50.,
                'generic_target_short_max_mm':90.,'strict_production':True}
        for w,h in ((100.,60.),(110.,70.),(100.,60.)):
            polygons=[np.array([[x,0],[x+w/3,0],[x+w/3,h],[x,h]],float)
                      for x in (0,w/3,2*w/3)]
            solution=solver.solve_generic_rectangle(polygons,config)
            np.testing.assert_allclose(solution.target_size_mm,[w,h],atol=1e-4)
            self.assertFalse(solution.relaxed_override)
            for pose in solution.poses:self.assertTrue(solver.is_rigid_transform(pose.transform_3x3))

    def test_assessment_endpoint_error_equals_independent_edge_transforms(self):
        polygons=[np.array([[0,0],[30,0],[30,60],[0,60]],float),
                  np.array([[30,0],[100,0],[100,60],[30,60]],float)]
        matches=[solver.EdgeMatch(0,1,1,3,0.,0.)]
        poses={0:solver.rigid_matrix(.123,[10,8]),1:solver.rigid_matrix(.123,[10.2,8.4])}
        candidate=solver._AssemblyCandidate(poses,frozenset({(0,1),(1,3)}),matches)
        expected=solver._match_endpoint_error(polygons,poses,matches[0])
        assessment=solver._assess_assembly(candidate,polygons,{})
        self.assertAlmostEqual(assessment.evaluation.endpoint_error_mm,expected,places=10)


if __name__=='__main__':unittest.main()
