"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import math

import torch
import torch.nn.functional as F

from src.quantization.transform import Transform

# TODO refactor:
#  the following 2 don't really fit the transform interface (we need to pass attention_mask just to fit input to output)

class EmbdsBatchWindows(Transform):
    """
    convert a batch of embds (shaped [batch_size, max_tokens_per_embd, token_size])
    to a batch of windows (of window_size)

    if max_tokens_per_embd * token_size isn't divisible by window_size the embd is padded with zeros

    (I'm only using torch.view and avoiding '-1' dimensions on purpose, to make sure the dimension are as expected)
    """

    def __init__(self, max_tokens_per_embd, token_size, window_size, batch_transform):
        embd_size = max_tokens_per_embd * token_size
        num_windows_per_embd = math.ceil(embd_size / window_size)
        embd_size_padded_to_window = num_windows_per_embd * window_size

        self.max_tokens_per_embd = max_tokens_per_embd
        self.token_size = token_size
        self.window_size = window_size
        self.batch_transform = batch_transform

        self.embd_size = embd_size
        self.num_windows_per_embd = num_windows_per_embd
        self.embd_size_padded_to_window = embd_size_padded_to_window

    def forward(self, x):
        embds_batch, attention_mask = x

        orig_batch_size = embds_batch.shape[0]

        # pad so that every embds has num_windows_per_doc (divides by window_size)
        embds_batch = F.pad(embds_batch.view(orig_batch_size, self.embd_size),
                            (0, self.embd_size_padded_to_window - self.embd_size))

        # view so we have a batch of windows
        embds_batch = embds_batch.view(orig_batch_size * self.num_windows_per_embd, self.window_size)

        # apply the inner batch-aware transform
        inner_f_res = self.batch_transform.forward(embds_batch)

        return (inner_f_res, attention_mask), orig_batch_size

    def backward(self, embds_batch_res):
        (inner_f_res, attention_mask), orig_batch_size = embds_batch_res

        windows_batch = self.batch_transform.backward(inner_f_res)
        # remove padding
        windows_batch = windows_batch.view(orig_batch_size, self.embd_size_padded_to_window)[:, :self.embd_size]
        # view as a batch of embds
        return windows_batch.view(orig_batch_size, self.max_tokens_per_embd, self.token_size), attention_mask


class EmbdsBatchLoop(Transform):
    def __init__(self, max_tokens_per_embd, vec_transform):
        self.max_tokens_per_embd = max_tokens_per_embd
        self.vec_transform = vec_transform

    def forward(self, x):
        embds_batch, attention_mask = x
        embds_lengths = attention_mask.sum(dim=1)

        return ([self.vec_transform.forward(d[:l])
                for d, l in zip(embds_batch, embds_lengths)], attention_mask), embds_lengths

    def backward(self, embds_batch_res):
        (inner_f_res_lst, attention_mask), embds_lengths = embds_batch_res

        return torch.stack([
            torch.nn.functional.pad(
                self.vec_transform.backward(inner_f_res),
                (0, 0, 0, self.max_tokens_per_embd - l)
            )
            for inner_f_res, l in zip(inner_f_res_lst, embds_lengths)
        ]), attention_mask
