# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import unittest
from copy import copy
import cv2
import torch
from fvcore.common.benchmark import benchmark
from torch.nn import functional as F

from detectron2.layers.roi_align import ROIAlign, roi_align


class ROIAlignTest(unittest.TestCase):
    def test_forward_output(self):
        input = np.arange(25).reshape(5, 5).astype("float32")
        """
        0  1  2   3 4
        5  6  7   8 9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        """

        output = self._simple_roialign(input, [1, 1, 3, 3], (4, 4), aligned=False)
        output_correct = self._simple_roialign(input, [1, 1, 3, 3], (4, 4), aligned=True)

        # without correction:
        old_results = [
            [7.5, 8, 8.5, 9],
            [10, 10.5, 11, 11.5],
            [12.5, 13, 13.5, 14],
            [15, 15.5, 16, 16.5],
        ]

        # with 0.5 correction:
        correct_results = [
            [4.5, 5.0, 5.5, 6.0],
            [7.0, 7.5, 8.0, 8.5],
            [9.5, 10.0, 10.5, 11.0],
            [12.0, 12.5, 13.0, 13.5],
        ]
        # This is an upsampled version of [[6, 7], [11, 12]]

        self.assertTrue(np.allclose(output.flatten(), np.asarray(old_results).flatten()))
        self.assertTrue(
            np.allclose(output_correct.flatten(), np.asarray(correct_results).flatten())
        )

        # Also see similar issues in tensorflow at
        # https://github.com/tensorflow/tensorflow/issues/26278

    def test_resize(self):
        H, W = 30, 30
        input = np.random.rand(H, W).astype("float32") * 100
        box = [10, 10, 20, 20]
        output = self._simple_roialign(input, box, (5, 5), aligned=True)

        input2x = cv2.resize(input, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
        box2x = [x / 2 for x in box]
        output2x = self._simple_roialign(input2x, box2x, (5, 5), aligned=True)
        diff = np.abs(output2x - output)
        self.assertTrue(diff.max() < 1e-4)

    def test_grid_sample_equivalence(self):
        H, W = 30, 30
        input = np.random.rand(H, W).astype("float32") * 100
        box = [10, 10, 20, 20]
        for ratio in [1, 2, 3]:
            output = self._simple_roialign(input, box, (5, 5), sampling_ratio=ratio)
            output_grid_sample = grid_sample_roi_align(
                torch.from_numpy(input[None, None, :, :]).float(),
                torch.as_tensor(box).float()[None, :],
                5,
                1.0,
                ratio,
            )
            self.assertTrue(torch.allclose(output, output_grid_sample))

    def _simple_roialign(self, img, box, resolution, sampling_ratio=0, aligned=True):
        """
        RoiAlign with scale 1.0.
        """
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        op = ROIAlign(resolution, 1.0, sampling_ratio, aligned=aligned)
        input = torch.from_numpy(img[None, None, :, :].astype("float32"))

        rois = [0] + list(box)
        rois = torch.from_numpy(np.asarray(rois)[None, :].astype("float32"))
        output = op.forward(input, rois)
        if torch.cuda.is_available():
            output_cuda = op.forward(input.cuda(), rois.cuda()).cpu()
            self.assertTrue(torch.allclose(output, output_cuda))
        return output[0, 0]

    def _simple_roialign_with_grad(self, img, box, resolution, device):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        op = ROIAlign(resolution, 1.0, 0, aligned=True)
        input = torch.from_numpy(img[None, None, :, :].astype("float32"))

        rois = [0] + list(box)
        rois = torch.from_numpy(np.asarray(rois)[None, :].astype("float32"))
        input = input.to(device=device)
        rois = rois.to(device=device)
        input.requires_grad = True
        output = op.forward(input, rois)
        return input, output

    def test_empty_box(self):
        img = np.random.rand(5, 5)
        box = [3, 4, 5, 4]
        o = self._simple_roialign(img, box, 7)
        self.assertTrue(o.shape == (7, 7))
        self.assertTrue((o == 0).all())

        for dev in ["cpu"] + ["cuda"] if torch.cuda.is_available() else []:
            input, output = self._simple_roialign_with_grad(img, box, 7, torch.device(dev))
            output.sum().backward()
            self.assertTrue(torch.allclose(input.grad, torch.zeros_like(input)))

    def test_empty_batch(self):
        input = torch.zeros(0, 3, 10, 10, dtype=torch.float32)
        rois = torch.zeros(0, 5, dtype=torch.float32)
        op = ROIAlign((7, 7), 1.0, 0, aligned=True)
        output = op.forward(input, rois)
        self.assertTrue(output.shape == (0, 3, 7, 7))


def grid_sample_roi_align(input, boxes, output_size, scale, sampling_ratio):
    # unlike true roi_align, this does not support different batch_idx
    from detectron2.projects.point_rend.point_features import (
        generate_regular_grid_point_coords,
        get_point_coords_wrt_image,
        point_sample,
    )

    N, _, H, W = input.shape
    R = len(boxes)
    assert N == 1
    boxes = boxes * scale
    grid = generate_regular_grid_point_coords(R, output_size * sampling_ratio, device=boxes.device)
    coords = get_point_coords_wrt_image(boxes, grid)
    coords = coords / torch.as_tensor([W, H], device=coords.device)  # R, s^2, 2
    res = point_sample(input, coords.unsqueeze(0), align_corners=False)  # 1,C, R,s^2
    res = (
        res.squeeze(0)
        .permute(1, 0, 2)
        .reshape(R, -1, output_size * sampling_ratio, output_size * sampling_ratio)
    )
    res = F.avg_pool2d(res, sampling_ratio)
    return res


def benchmark_roi_align():
    def random_boxes(mean_box, stdev, N, maxsize):
        ret = torch.rand(N, 4) * stdev + torch.tensor(mean_box, dtype=torch.float)
        ret.clamp_(min=0, max=maxsize)
        return ret

    def func(shape, nboxes_per_img, sampling_ratio, device, box_size="large"):
        N, _, H, _ = shape
        input = torch.rand(*shape)
        boxes = []
        batch_idx = []
        for k in range(N):
            if box_size == "large":
                b = random_boxes([80, 80, 130, 130], 24, nboxes_per_img, H)
            else:
                b = random_boxes([100, 100, 110, 110], 4, nboxes_per_img, H)
            boxes.append(b)
            batch_idx.append(torch.zeros(nboxes_per_img, 1, dtype=torch.float32) + k)
        boxes = torch.cat(boxes, axis=0)
        batch_idx = torch.cat(batch_idx, axis=0)
        boxes = torch.cat([batch_idx, boxes], axis=1)

        input = input.to(device=device)
        boxes = boxes.to(device=device)

        def bench():
            if False and sampling_ratio > 0 and N == 1:
                # enable to benchmark grid_sample (slower)
                grid_sample_roi_align(input, boxes[:, 1:], 7, 1.0, sampling_ratio)
            else:
                roi_align(input, boxes, 7, 1.0, sampling_ratio, True)
            if device == "cuda":
                torch.cuda.synchronize()

        return bench

    def gen_args(arg):
        args = []
        for size in ["small", "large"]:
            for ratio in [0, 2]:
                args.append(copy(arg))
                args[-1]["sampling_ratio"] = ratio
                args[-1]["box_size"] = size
        return args

    arg = dict(shape=(1, 512, 256, 256), nboxes_per_img=512, device="cuda")
    benchmark(func, "cuda_roialign", gen_args(arg), num_iters=20, warmup_iters=1)
    arg.update({"device": "cpu", "shape": (1, 256, 128, 128)})
    benchmark(func, "cpu_roialign", gen_args(arg), num_iters=5, warmup_iters=1)


if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_roi_align()
    unittest.main()
