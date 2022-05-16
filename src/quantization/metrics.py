"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch


def SS(vec):
    """:return: sum of squares"""
    return torch.sum(vec ** 2)


def estimate_vNMSE(vec, tensor_transform, repeats=5):
    """:return: vec's normalized mean squared error"""
    SSE_accum = 0
    for _ in range(repeats):
        reconstructed_vec = tensor_transform.roundtrip(vec.clone())

        SSE_accum += SS(vec - reconstructed_vec)

    return (SSE_accum / repeats) / SS(vec)



def estimate_vSSE(vec, tensor_transform, repeats=5):
    """:return: vec's mean squared error"""
    SSE_accum = 0
    for _ in range(repeats):
        reconstructed_vec = tensor_transform.roundtrip(vec.clone())

        SSE_accum += SS(vec - reconstructed_vec)

    return (SSE_accum / repeats)
