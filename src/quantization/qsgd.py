"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from src.quantization.transform import Transform


def _dither_like(x, seed):
    return torch.empty_like(x).uniform_(-0.5, 0.5,
                                        generator=torch.Generator(device=x.device).manual_seed(seed))


class QSGD(Transform):
    def __init__(self, gen_seed=torch.seed, bits=2, norm_ord=float('inf')):
        self.gen_seed = gen_seed
        self.norm_ord = norm_ord

        self.max_val = 2 ** (bits - 1) - 1  # one bit is used for sign

    def forward(self, x):
        """
            :param x: any tensor that can be normalized by normalization_factory into [0, 2**bits-1] range
            :return: a tensor with the same dtype in {0, ... , 2**bits - 1},
                     and a random seed for the dither noise if 'subtractive' is True
        """
        x_norm = torch.linalg.norm(x, ord=self.norm_ord, dim=1, keepdim=True)
        abs_x = x.abs()
        sign_x = x.sign()

        normalized_x = torch.nan_to_num(abs_x * (self.max_val / x_norm))

        seed = self.gen_seed()

        return torch.round(normalized_x + _dither_like(x, seed)), (sign_x, x_norm)

    def backward(self, f_res):
        x, (sign_x, x_norm) = f_res

        return sign_x * (x / self.max_val) * x_norm
