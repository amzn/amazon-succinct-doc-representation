"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
# Taken from 'DRIVE: One-bit Distributed Mean Estimation' paper

import math

import torch
import torch.nn.functional as F

from src.quantization.stochastic_quantization import StochasticQuantization
from src.quantization.transform import Transform, OnionTransform


def hadamard_vec_(vec):
    """
    fast Walsh–Hadamard transform (in-place)

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    numel = vec.numel()

    h = 2
    while h <= numel:
        hf = h // 2
        vec = vec.view(numel // h, h)

        ## the following is a more inplace way of doing the following:
        # half_1 = batch[:, :, :hf]
        # half_2 = batch[:, :, hf:]
        # batch = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
        # the NOT inplace version is actually slightly faster
        # (I assume for making more memory-contiguous operations. That being said,
        # it more easily throws out-of-memory and may slow things overall,
        # so using mostly inplace version below:)

        vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
        vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
        h *= 2

    vec *= numel ** -0.5  # divide by sqrt(numel)


@torch.jit.script
def hadamard_batch(batch):
    """
    vectorized, NOT in-place, fast Walsh–Hadamard transform

    :param batch: a batch of vectors each is expected to have floating point dtype and power of 2 elements!
    :return: the Hadamard transform of each vector
    """
    batch_dim, vec_dim = batch.shape

    h = 2
    while h <= vec_dim:
        hf = h // 2
        batch = batch.view(batch_dim, vec_dim // h, h)

        half_1 = batch[:, :, :hf]
        half_2 = batch[:, :, hf:]
        batch = torch.cat(
            (
                half_1 + half_2,
                half_1 - half_2
            ),
            dim=-1
        )
        h *= 2

    batch *= vec_dim ** -0.5  # divide by sqrt(vec_dim)

    return batch.view(batch_dim, vec_dim)


def hadamard(x, batched):
    if batched:
        x = hadamard_batch(x)
    else:
        hadamard_vec_(x)
    return x


def rademacher_like(x, generator):
    return 2 * torch.torch.empty_like(x).bernoulli_(generator=generator) - 1


def next_power_of_2(n):
    return 2 ** (math.floor(math.log2(n)) + 1)


def pad_to_next_power_of_2(x):
    numel = x.numel()

    numel_after_pad = next_power_of_2(numel)
    return F.pad(x, (0, numel_after_pad - numel))


class PadToPowerOf2(Transform):
    def forward(self, x):
        """
        :param x: assumes vec is 1d
        :return: x padded with zero until the next power-of-2
        """
        numel = x.numel()
        # pad to nearest power of 2 if needed
        if numel & (numel - 1) != 0:
            return pad_to_next_power_of_2(x), numel
        else:
            return x, numel

    def backward(self, f_res):
        x, original_numel = f_res
        return x[:original_numel]


# singleton
PadToPowerOf2.s = PadToPowerOf2()


class Flatten(Transform):
    def forward(self, x):
        return x.view(-1), x.shape

    def backward(self, f_res):
        x, original_shape = f_res
        return x.view(original_shape)


# singleton
Flatten.s = Flatten()


class RandomizedHadamard(Transform):
    def __init__(self, gen_seed, batched=False):
        self.gen_seed = gen_seed
        self.batched = batched

    def forward(self, x):
        seed = self.gen_seed()

        d = rademacher_like(x, torch.Generator(device=x.device).manual_seed(seed))

        return hadamard(x * d, self.batched), seed

    def backward(self, f_res):
        x, seed = f_res

        d = rademacher_like(x, torch.Generator(device=x.device).manual_seed(seed))

        return hadamard(x, self.batched) * d


class RotatedQuantization(OnionTransform):
    def __init__(self, pre_rotation_flat_and_pad, rotation, post_rotation_transform):
        if pre_rotation_flat_and_pad:
            transforms = [Flatten.s, PadToPowerOf2.s]
        else:
            transforms = []
        transforms += [rotation, post_rotation_transform]

        super().__init__(transforms)


class HadamardSQ(RotatedQuantization):
    def __init__(self, gen_seed, bits, subtractive=False, batched=False):
        super().__init__(
            pre_rotation_flat_and_pad=not batched,
            rotation=RandomizedHadamard(gen_seed, batched),
            post_rotation_transform=StochasticQuantization(bits, subtractive, batched),
        )
