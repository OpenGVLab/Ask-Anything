# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import unittest
import torch

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.testing import random_boxes

logger = logging.getLogger(__name__)


class TestROIPooler(unittest.TestCase):
    def _test_roialignv2_roialignrotated_match(self, device):
        pooler_resolution = 14
        canonical_level = 4
        canonical_scale_factor = 2 ** canonical_level
        pooler_scales = (1.0 / canonical_scale_factor,)
        sampling_ratio = 0

        N, C, H, W = 2, 4, 10, 8
        N_rois = 10
        std = 11
        mean = 0
        feature = (torch.rand(N, C, H, W) - 0.5) * 2 * std + mean

        features = [feature.to(device)]

        rois = []
        rois_rotated = []
        for _ in range(N):
            boxes = random_boxes(N_rois, W * canonical_scale_factor)
            rotated_boxes = torch.zeros(N_rois, 5)
            rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
            rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
            rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            rois.append(Boxes(boxes).to(device))
            rois_rotated.append(RotatedBoxes(rotated_boxes).to(device))

        roialignv2_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        roialignv2_out = roialignv2_pooler(features, rois)

        roialignrotated_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignRotated",
        )

        roialignrotated_out = roialignrotated_pooler(features, rois_rotated)

        self.assertTrue(torch.allclose(roialignv2_out, roialignrotated_out, atol=1e-4))

    def test_roialignv2_roialignrotated_match_cpu(self):
        self._test_roialignv2_roialignrotated_match(device="cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_roialignv2_roialignrotated_match_cuda(self):
        self._test_roialignv2_roialignrotated_match(device="cuda")

    def _test_scriptability(self, device):
        pooler_resolution = 14
        canonical_level = 4
        canonical_scale_factor = 2 ** canonical_level
        pooler_scales = (1.0 / canonical_scale_factor,)
        sampling_ratio = 0

        N, C, H, W = 2, 4, 10, 8
        N_rois = 10
        std = 11
        mean = 0
        feature = (torch.rand(N, C, H, W) - 0.5) * 2 * std + mean

        features = [feature.to(device)]

        rois = []
        for _ in range(N):
            boxes = random_boxes(N_rois, W * canonical_scale_factor)

            rois.append(Boxes(boxes).to(device))

        roialignv2_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        roialignv2_out = roialignv2_pooler(features, rois)
        scripted_roialignv2_out = torch.jit.script(roialignv2_pooler)(features, rois)
        self.assertTrue(torch.equal(roialignv2_out, scripted_roialignv2_out))

    def test_scriptability_cpu(self):
        self._test_scriptability(device="cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptability_gpu(self):
        self._test_scriptability(device="cuda")

    def test_no_images(self):
        N, C, H, W = 0, 32, 32, 32
        feature = torch.rand(N, C, H, W) - 0.5
        features = [feature]
        pooler = ROIPooler(
            output_size=14, scales=(1.0,), sampling_ratio=0.0, pooler_type="ROIAlignV2"
        )
        output = pooler.forward(features, [])
        self.assertEqual(output.shape, (0, C, 14, 14))

    def test_roi_pooler_tracing(self):
        class Model(torch.nn.Module):
            def __init__(self, roi):
                super(Model, self).__init__()
                self.roi = roi

            def forward(self, x, boxes):
                return self.roi(x, [Boxes(boxes)])

        pooler_resolution = 14
        canonical_level = 4
        canonical_scale_factor = 2 ** canonical_level
        pooler_scales = (1.0 / canonical_scale_factor, 0.5 / canonical_scale_factor)
        sampling_ratio = 0

        N, C, H, W = 1, 4, 10, 8
        N_rois = 10
        std = 11
        mean = 0
        feature = (torch.rand(N, C, H, W) - 0.5) * 2 * std + mean
        feature = [feature, feature]

        rois = random_boxes(N_rois, W * canonical_scale_factor)
        # Add one larger box so that this level has only one box.
        # This may trigger the bug https://github.com/pytorch/pytorch/issues/49852
        # that we shall workaround.
        rois = torch.cat([rois, torch.tensor([[0, 0, 448, 448]])])

        model = Model(
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type="ROIAlign",
            )
        )

        with torch.no_grad():
            func = torch.jit.trace(model, (feature, rois))
            o = func(feature, rois)
            self.assertEqual(o.shape, (11, 4, 14, 14))
            o = func(feature, rois[:5])
            self.assertEqual(o.shape, (5, 4, 14, 14))
            o = func(feature, random_boxes(20, W * canonical_scale_factor))
            self.assertEqual(o.shape, (20, 4, 14, 14))


if __name__ == "__main__":
    unittest.main()
