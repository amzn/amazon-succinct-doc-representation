"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
from src.quantization.metrics import SS
from src.quantization.transform import Transform


class RepeatTransform(Transform):
    def __init__(self, inner, repeats=10):
        self.inner = inner
        self.repeats = repeats

    def forward(self, x):
        best_SSE = float('inf')
        for _ in range(self.repeats):
            inner_res = self.inner.forward(x.clone())
            reconstructed_x = self.inner.backward(inner_res)
            SSE = SS(x - reconstructed_x)
            if SSE < best_SSE:
                best_inner_res = inner_res
                best_SSE = SSE
        return best_inner_res

    def backward(self, f_res):
        return self.inner.backward(f_res)
