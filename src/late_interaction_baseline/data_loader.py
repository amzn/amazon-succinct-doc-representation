"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import pandas as pd
import transformers
from tqdm.contrib import tzip
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import sys
from tqdm.auto import tqdm

from src.config import ARTIFACTS_PATH, DOC_MAX_LENGTH

print("loaded", sys.modules[__name__])
"""
Input to the naive late interaction model is 24 words from query 
and 172 words from doc, without dynamic padding (using DOC_MAX_LENGTH as parameter for doc size).  
"""


class TsvDatasetSep(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, dfiter=None):
        self.dfiter = dfiter
        if dfiter is None:
            self.dfiter = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/triples.train.sample.tsv", sep="\t",
                                      chunksize=10000, names=["query", "pos", "neg"])
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer

    def __iter__(self):
        for df in tqdm(self.dfiter):
            queries = [x for x in df["query"] for _ in range(2)]
            passages = [val for x in zip(df["pos"], df["neg"]) for val in x]
            if "pos_score" not in df:
                labels = [1, 0] * len(df)
            else:  # kd settings
                labels = [score for pair in zip(df["pos_score"], df["neg_score"]) for score in pair]
            q_tokens = self.tokenizer(queries, max_length=24, truncation=TruncationStrategy.LONGEST_FIRST,
                                      padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")
            d_tokens = self.tokenizer(passages, max_length=DOC_MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST,
                                      padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")
            # document is the second piece of text, even though it is encoded as a single text. Therefore, type_id is 1
            d_tokens["token_type_ids"][:, :] = 1
            tokens = {k: torch.cat((q_tokens[k], d_tokens[k]), dim=1) for k in q_tokens.keys()}
            for i in range(len(labels)):
                yield {**{k: v[i] for k, v in tokens.items()}, "label": labels[i]}


class TsvDatasetPredictSep(torch.utils.data.IterableDataset):
    def __init__(self, df, tokenizer, label_field_name="label"):
        tokenizer: transformers.PreTrainedTokenizer = tokenizer
        queries = df["query"].tolist()
        passages = df["doc"].tolist()
        labels = df["label"].tolist()
        self.label_field_name = label_field_name  # Hack: sometimes transformers likes "label" and sometimes "labels"
        q_tokens = tokenizer(queries, max_length=24, truncation=TruncationStrategy.LONGEST_FIRST,
                             padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")
        d_tokens = tokenizer(passages, max_length=DOC_MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST,
                             padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")
        # document is the second piece of text, even though it is encoded as a single text. Therefore, type_id is 1
        d_tokens["token_type_ids"][:, :] = 1
        tokens = {k: torch.cat((q_tokens[k], d_tokens[k]), dim=1) for k in q_tokens.keys()}

        self.tokens = tokens
        self.labels = labels

    def __iter__(self):
        for label, tokens_tuple in tzip(self.labels, zip(*self.tokens.values())):
            yield {
                **dict(zip(self.tokens.keys(), tokens_tuple)),
                self.label_field_name: label,
            }
