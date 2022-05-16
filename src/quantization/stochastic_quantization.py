"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from src.quantization.normalization import MinMaxNormalization
from src.quantization.transform import Transform


def _dither_like(x, seed):
    return torch.empty_like(x).uniform_(-0.5, 0.5,
                                        generator=torch.Generator(device=x.device).manual_seed(seed))


class StochasticQuantization(Transform):
    def __init__(self, bits, subtractive=False, batched=False,
                 normalization_factory=MinMaxNormalization,
                 gen_seed=torch.seed):
        self.normalization = normalization_factory(bits, batched)
        self.subtractive = subtractive
        self.gen_seed = gen_seed

    def forward(self, x):
        """
            :param x: any tensor that can be normalized by normalization_factory into [0, 2**bits-1] range
            :return: a tensor with the same dtype in {0, ... , 2**bits - 1},
                     and a random seed for the dither noise if 'subtractive' is True
        """
        x, normalization_context = self.normalization.forward(x)

        seed = self.gen_seed()

        return torch.round(x + _dither_like(x, seed)), (seed, normalization_context)

    def backward(self, f_res):
        x, (seed, normalization_context) = f_res
        if self.subtractive:
            x = x - _dither_like(x, seed)

        return self.normalization.backward((x, normalization_context))
