# Copyright (c) 2024 Ming-Yang Ho, Che-Ming Wu, and Min-Sheng Wu
# All rights reserved.
#
# This source code is adapted for UNSB integration.

import torch


class Interpolation3D:
    """Bilinear interpolation from 3x3 stats grid to HxW pixel grid."""

    def __init__(self, channel):
        self.channel = channel
        self.is_init = False
        self.device = None
        self.size = None
        self.half_size = None
        self.eps = 1e-7

    def init(self, size, device=None):
        if self.is_init and self.size == size and (device is None or device == self.device):
            return

        self.size = size
        self.half_size = size // 2
        if device is None:
            device = self.device if self.device is not None else torch.device('cpu')
        self.device = device
        self.init_matrix()
        self.is_init = True

    def init_matrix(self):
        self.small_to_large = torch.arange(0.5, self.size + 0.5, 1, device=self.device)
        self.large_to_small = torch.arange(self.size - 0.5, 0, -1, device=self.device)

        self.top_left = (self.large_to_small * self.large_to_small.unsqueeze(0).T) / self.size / self.size
        self.down_left = (self.large_to_small * self.small_to_large.unsqueeze(0).T) / self.size / self.size
        self.top_right = (self.small_to_large * self.large_to_small.unsqueeze(0).T) / self.size / self.size
        self.down_right = (self.small_to_large * self.small_to_large.unsqueeze(0).T) / self.size / self.size

        self.top_left = self.top_left.contiguous()
        self.down_left = self.down_left.contiguous()
        self.top_right = self.top_right.contiguous()
        self.down_right = self.down_right.contiguous()

    def top_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return (
            top_left_value * self.top_left[-self.half_size:, -self.half_size:]
            + top_right_value * self.top_right[-self.half_size:, -self.half_size:]
            + down_left_value * self.down_left[-self.half_size:, -self.half_size:]
            + down_right_value * self.down_right[-self.half_size:, -self.half_size:]
        )

    def top_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return (
            top_left_value * self.top_left[-self.half_size:, :self.half_size]
            + top_right_value * self.top_right[-self.half_size:, :self.half_size]
            + down_left_value * self.down_left[-self.half_size:, :self.half_size]
            + down_right_value * self.down_right[-self.half_size:, :self.half_size]
        )

    def down_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return (
            top_left_value * self.top_left[:self.half_size, -self.half_size:]
            + top_right_value * self.top_right[:self.half_size, -self.half_size:]
            + down_left_value * self.down_left[:self.half_size, -self.half_size:]
            + down_right_value * self.down_right[:self.half_size, -self.half_size:]
        )

    def down_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return (
            top_left_value * self.top_left[:self.half_size, :self.half_size]
            + top_right_value * self.top_right[:self.half_size, :self.half_size]
            + down_left_value * self.down_left[:self.half_size, :self.half_size]
            + down_right_value * self.down_right[:self.half_size, :self.half_size]
        )

    def _interpolation_mean_table(
        self,
        y0x0,
        y0x1,
        y0x2,
        y1x0,
        y1x1,
        y1x2,
        y2x0,
        y2x1,
        y2x2,
    ):
        table = torch.zeros((self.channel, self.size, self.size), device=self.device)
        table[:, :self.half_size, :self.half_size] = self.top_left_corner(y0x0, y0x1, y1x0, y1x1)
        table[:, :self.half_size, self.half_size:] = self.top_right_corner(y0x1, y0x2, y1x1, y1x2)
        table[:, self.half_size:, :self.half_size] = self.down_left_corner(y1x0, y1x1, y2x0, y2x1)
        table[:, self.half_size:, self.half_size:] = self.down_right_corner(y1x1, y1x2, y2x1, y2x2)
        return table

    def deal_with_inf(self, matrix_3x3):
        return torch.where(
            torch.logical_or(torch.isinf(matrix_3x3), torch.isnan(matrix_3x3)),
            matrix_3x3[:, 1:2, 1:2],
            matrix_3x3,
        )

    def interpolation_mean_table(self, matrix_3x3):
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )

    def interpolation_std_table_inverse(self, matrix_3x3):
        """Deprecated: 先取倒数再插值，数学上有问题，保留用于兼容"""
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)
        matrix_3x3 = 1 / (matrix_3x3 + self.eps)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )

    def interpolation_std_table(self, matrix_3x3):
        """正确方式：先插值std，调用方再取倒数"""
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )
