"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from src.quantization.transform import Transform


class MinMaxNormalization(Transform):
    def __init__(self, bits, batched=False):
        self.max_val = 2 ** bits - 1
        if batched:
            self.op_kwargs = {"dim": 1, "keepdim": True}
        else:
            self.op_kwargs = {"dim": 0}

    def forward(self, x):
        minimum = x.min(**self.op_kwargs).values
        maximum = x.max(**self.op_kwargs).values

        return torch.nan_to_num((x - minimum) * (self.max_val / (maximum - minimum))), (minimum, maximum)

    def backward(self, f_res):
        x, (minimum, maximum) = f_res
        return (x / self.max_val) * (maximum - minimum) + minimum
