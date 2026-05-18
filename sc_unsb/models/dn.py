# Copyright (c) 2024 Ming-Yang Ho, Che-Ming Wu, and Min-Sheng Wu
# All rights reserved.
#
# This source code is adapted for UNSB integration.

import typing as t

import torch
import torch.nn as nn

from .interpolation import Interpolation3D


class DenseInstanceNorm(nn.Module):
    def __init__(self, out_channels: int, affine: bool = False):
        super().__init__()

        # if use normal instance normalization during evaluation mode
        self.normal_instance_normalization = False

        # if collecting instance normalization mean and std during evaluation mode
        self.collection_mode = False

        self.out_channels = out_channels
        self.interpolation3d = Interpolation3D(channel=out_channels)
        self.eps = 1e-6

        if affine:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # device anchor
        self.register_buffer('_dn_dummy', torch.empty(0), persistent=False)

    def _device(self):
        return self._dn_dummy.device

    def init_collection(self, y_anchor_num: int, x_anchor_num: int, num_timesteps: int = 1) -> None:
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.num_timesteps = num_timesteps
        self.mean_table = torch.zeros(num_timesteps, y_anchor_num, x_anchor_num, self.out_channels, device=self._device())
        self.std_table = torch.zeros(num_timesteps, y_anchor_num, x_anchor_num, self.out_channels, device=self._device())

    def pad_table(self, padding: int = 1) -> None:
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(self.mean_table.permute(0, 3, 1, 2))
        self.padded_std_table = pad_func(self.std_table.permute(0, 3, 1, 2))

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.clamp(x_std, min=self.eps)
        x = (x - x_mean) / x_std
        return x

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: t.Optional[int] = None,
        x_anchor: t.Optional[int] = None,
        padding: int = 1,
        time_idx: int = 0,
    ) -> torch.Tensor:
        if self.training or self.normal_instance_normalization:
            _, _, h, _ = x.shape
            self.interpolation3d.init(size=h, device=x.device)
            return self.forward_normal(x)

        if y_anchor is None or x_anchor is None:
            raise ValueError('y_anchor/x_anchor is None')

        if self.collection_mode:
            _, _, h, _ = x.shape
            self.interpolation3d.init(size=h, device=x.device)
            x_std, x_mean = torch.std_mean(x, dim=(2, 3))  # [B, C]
            x_std = torch.clamp(x_std, min=self.eps)
            self.mean_table[time_idx, y_anchor, x_anchor] = x_mean
            self.std_table[time_idx, y_anchor, x_anchor] = x_std
            x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
            x_std = x_std.unsqueeze(-1).unsqueeze(-1)
            x = (x - x_mean) / x_std
            if self.weight is not None:
                x = x * self.weight + self.bias
        else:
            if x.shape[0] != 1:
                raise ValueError('only support batch size = 1')

            _, _, h, _ = x.shape
            self.interpolation3d.init(size=h, device=x.device)

            top = y_anchor
            left = x_anchor
            down = y_anchor + 2 * padding + 1
            right = x_anchor + 2 * padding + 1

            x_mean = self.padded_mean_table[time_idx, :, top:down, left:right]
            x_std = self.padded_std_table[time_idx, :, top:down, left:right]

            x_mean = self.interpolation3d.interpolation_mean_table(x_mean).unsqueeze(0)
            x_std = self.interpolation3d.interpolation_std_table(x_std).unsqueeze(0)
            x_std = torch.clamp(x_std, min=self.eps)

            x = (x - x_mean) / x_std
            if self.weight is not None:
                x = x * self.weight + self.bias

        return x


class PrefetchDenseInstanceNorm(nn.Module):
    def __init__(self, out_channels: int, affine: bool = False):
        super().__init__()

        self.out_channels = out_channels
        self.interpolation3d = Interpolation3D(channel=out_channels)
        self.eps = 1e-6

        if affine:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('_dn_dummy', torch.empty(0), persistent=False)
        self.pad_func = nn.ReplicationPad2d((1, 1, 1, 1))

    def _device(self):
        return self._dn_dummy.device

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=self._device())
        self.std_table = torch.zeros(y_anchor_num, x_anchor_num, self.out_channels, device=self._device())
        self.pad_table()

    def pad_table(self, padding: int = 1) -> None:
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(self.mean_table.permute(2, 0, 1).unsqueeze(0))
        self.padded_std_table = pad_func(self.std_table.permute(2, 0, 1).unsqueeze(0))

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.clamp(x_std, min=self.eps)
        x = (x - x_mean) / x_std
        return x

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: int = None,
        x_anchor: int = None,
        padding: int = 1,
        pre_y_anchor: t.Optional[t.List[int]] = None,
        pre_x_anchor: t.Optional[t.List[int]] = None,
    ) -> torch.Tensor:
        n, _, h, _ = x.shape
        real_x, pre_x = torch.split(x, (1, n - 1), dim=0)

        self.interpolation3d.init(size=h, device=x.device)

        if pre_y_anchor is not None and pre_x_anchor is not None:
            pre_x_std, pre_x_mean = torch.std_mean(pre_x, dim=(2, 3))
            pre_x_std = torch.clamp(pre_x_std, min=self.eps)

            for i, (sub_pre_y_anchor, sub_pre_x_anchor) in enumerate(zip(pre_y_anchor, pre_x_anchor)):
                if sub_pre_y_anchor == -1:
                    continue
                self.mean_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_mean[i]
                self.std_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_std[i]

            pre_x_mean = pre_x_mean.unsqueeze(-1).unsqueeze(-1)
            pre_x_std = pre_x_std.unsqueeze(-1).unsqueeze(-1)

            pre_x = (pre_x - pre_x_mean) / pre_x_std
            if self.weight is not None:
                pre_x = pre_x * self.weight + self.bias

        if y_anchor != -1 and x_anchor != -1:
            top = y_anchor
            left = x_anchor
            down = y_anchor + 2 * padding + 1
            right = x_anchor + 2 * padding + 1

            self.pad_table()
            x_mean = self.padded_mean_table[:, :, top:down, left:right]
            x_std = self.padded_std_table[:, :, top:down, left:right]
            x_mean = x_mean.squeeze(0)
            x_std = x_std.squeeze(0)

            x_mean_expand = x_mean[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_std_expand = x_std[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_mean = torch.where(x_mean == 0, x_mean_expand, x_mean)
            x_std = torch.where(x_std == 0, x_std_expand, x_std)

            x_mean = self.interpolation3d.interpolation_mean_table(x_mean).unsqueeze(0)
            x_std = self.interpolation3d.interpolation_std_table(x_std).unsqueeze(0)
            x_std = torch.clamp(x_std, min=self.eps)

            real_x = (real_x - x_mean) / x_std
            if self.weight is not None:
                real_x = real_x * self.weight + self.bias

        x = torch.cat((real_x, pre_x), dim=0)
        return x


def not_use_dense_instance_norm(model: nn.Module) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True


def init_dense_instance_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int, num_timesteps: int = 1) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num, num_timesteps=num_timesteps)


def use_dense_instance_norm(model: nn.Module, padding: int = 1) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


def init_prefetch_dense_instance_norm(model: nn.Module, y_anchor_num: int, x_anchor_num: int) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, PrefetchDenseInstanceNorm):
            layer.init_collection(y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num)


def replace_instance_norm_with_dn(model: nn.Module) -> nn.Module:
    """
    Replace all InstanceNorm2d layers with DenseInstanceNorm layers.

    This enables plug-and-play DN inference for models trained with standard IN.
    Since IN and DN behave identically during training (both compute per-instance
    mean/std normalization), the replacement is mathematically equivalent.

    Parameters:
        model (nn.Module) - model with InstanceNorm2d layers

    Returns:
        nn.Module - model with DenseInstanceNorm layers (modified in-place)
    """
    # Get device from model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # Handle nn.Sequential - need to rebuild with replaced modules
    if isinstance(model, nn.Sequential):
        new_modules = []
        for module in model:
            if isinstance(module, nn.InstanceNorm2d):
                dn_layer = DenseInstanceNorm(
                    out_channels=module.num_features,
                    affine=module.affine
                ).to(device)
                if module.affine:
                    dn_layer.weight.data = module.weight.data.clone()
                    dn_layer.bias.data = module.bias.data.clone()
                new_modules.append(dn_layer)
            else:
                replace_instance_norm_with_dn(module)
                new_modules.append(module)
        # Replace the internal list
        model._modules.clear()
        for i, m in enumerate(new_modules):
            model.add_module(str(i), m)
        return model

    # Handle nn.ModuleList - replace in place
    if isinstance(model, nn.ModuleList):
        for i, module in enumerate(model):
            if isinstance(module, nn.InstanceNorm2d):
                dn_layer = DenseInstanceNorm(
                    out_channels=module.num_features,
                    affine=module.affine
                ).to(device)
                if module.affine:
                    dn_layer.weight.data = module.weight.data.clone()
                    dn_layer.bias.data = module.bias.data.clone()
                model[i] = dn_layer
            else:
                replace_instance_norm_with_dn(module)
        return model

    # Handle regular modules with named children
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d):
            # Create DN layer with same parameters
            dn_layer = DenseInstanceNorm(
                out_channels=module.num_features,
                affine=module.affine
            ).to(device)
            # Copy affine parameters if they exist
            if module.affine:
                dn_layer.weight.data = module.weight.data.clone()
                dn_layer.bias.data = module.bias.data.clone()
            setattr(model, name, dn_layer)
        else:
            # Recursively replace in child modules
            replace_instance_norm_with_dn(module)
    return model


def count_instance_norm_layers(model: nn.Module) -> int:
    """Count the number of InstanceNorm2d layers in a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.InstanceNorm2d):
            count += 1
    return count


def count_dn_layers(model: nn.Module) -> int:
    """Count the number of DenseInstanceNorm layers in a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, DenseInstanceNorm):
            count += 1
    return count
