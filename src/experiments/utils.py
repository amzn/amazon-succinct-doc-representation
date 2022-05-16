"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
from time import time

from src.config import DOC_MAX_LENGTH
from src.quantization.embds_batch import EmbdsBatchWindows

WINDOW_SIZE = 128


def print_times(func):
    def wrap(*args, **kwargs):
        print(f'start {func.__name__}')
        start = time()
        result = func(*args, **kwargs)
        print(f'end   {func.__name__} {time() - start:.2f} seconds')
        return result

    return wrap


def as_batch(ae_num_features, batch_transform, max_tokens_per_embd=DOC_MAX_LENGTH):
    return EmbdsBatchWindows(
        max_tokens_per_embd=max_tokens_per_embd,
        token_size=ae_num_features,
        window_size=WINDOW_SIZE,
        batch_transform=batch_transform
    )
