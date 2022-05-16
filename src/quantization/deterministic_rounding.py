"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from src.quantization.normalization import MinMaxNormalization
from src.quantization.rotated_quantization import RotatedQuantization, RandomizedHadamard
from src.quantization.transform import Transform, OnionTransform


class Round(Transform):
  def forward(self, x):
    return torch.round(x), []

  def backward(self, f_res):
    x, _ = f_res
    return x


# singleton
Round.s = Round()


class DeterministicRounding(OnionTransform):
  def __init__(self, bits, batched=False, normalization_factory=MinMaxNormalization):
    super().__init__([normalization_factory(bits, batched), Round.s])


class HadamardDR(RotatedQuantization):
  def __init__(self, gen_seed, bits, batched=False):
    super().__init__(
      pre_rotation_flat_and_pad=not batched,
      rotation=RandomizedHadamard(gen_seed, batched),
      post_rotation_transform=DeterministicRounding(bits, batched),
    )
