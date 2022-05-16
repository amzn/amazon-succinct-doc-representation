"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from src.quantization.transform import Transform


class Float16Compressor(Transform):
    def forward(self, x):
        return x.to(torch.float16), x.dtype

    def backward(self, f_res):
        # todo normalize the Transform API so this kind of unpacking could be avoided
        x, dtype = f_res
        return x.to(dtype)


# singleton
Float16Compressor.s = Float16Compressor()