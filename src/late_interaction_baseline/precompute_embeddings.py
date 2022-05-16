"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import pickle
from collections import namedtuple

import pandas as pd
import torch
import transformers
from tqdm.auto import trange
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from src.config import ARTIFACTS_PATH, CACHED_EMBDS_PATH, DOC_MAX_LENGTH
from src.late_interaction_baseline.data_loader import TsvDatasetPredictSep
from src.late_interaction_baseline.modeling import E2ELate


def sample_collection():
    df = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/collection.tsv", sep="\t", names=["docid", "doc"])
    df = df.sample(frac=1, random_state=42)
    df[:1000000].to_csv(f"{ARTIFACTS_PATH}/msmarco/sample_collection.csv", index=False)


def get_embeddings_path(embd_type: str, prefix: str):
    return f"{CACHED_EMBDS_PATH}/{prefix}_{embd_type}_embeddings.pkl"


def generate_embeddings(data_path: str, embd_type: str, prefix: str, e2e_late: E2ELate = None,
                        e2e_late_path: str = None):
    """
    :param embd_type: 'doc' | 'query'
    """
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    if e2e_late is None:
        e2e_late: E2ELate = E2ELate.from_pretrained(e2e_late_path)
    if embd_type == 'doc':
        model = e2e_late.doc_encoder.cuda()
        max_length = DOC_MAX_LENGTH
    else:  # embd_type == query
        model = e2e_late.query_encoder.cuda()
        max_length = 24

    model.eval()

    df = pd.read_csv(data_path)
    passages = df[embd_type].tolist()

    with open(get_embeddings_path(embd_type, prefix), "wb") as handle:
        for i in trange(0, len(df), 32):
            tokens = tokenizer(passages[i:i + 32], max_length=max_length,
                               truncation=TruncationStrategy.LONGEST_FIRST,
                               padding=PaddingStrategy.LONGEST, return_tensors="pt")
            if embd_type == 'doc':
                tokens["token_type_ids"][:, :] = 1  # remove for query embeddings

            tokens = {k: v.cuda() for k, v in tokens.items()}
            with torch.no_grad():
                preds = model(**tokens)
            preds = preds.last_hidden_state.cpu().numpy()
            lengths = tokens["attention_mask"].sum(dim=1).cpu()
            vecs = [v[:l] for v, l in zip(preds, lengths)]
            pickle.dump(vecs, handle)


def generate_embeddings_from_model(data_path: str, prefix: str):
    # the doc / query generate_embeddings seemed to produce a slightly different vectors
    # this methods tries to directly use the E2ELate the same way it is used in e2e_predict.py
    e2e_late: E2ELate = E2ELate.from_pretrained(f"{ARTIFACTS_PATH}/msmarco/output/naive_late_65k/",
                                                save_embds=True,
                                                saved_embds_prefix=prefix,
                                                )
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    df = pd.read_csv(data_path)

    args = transformers.TrainingArguments(output_dir="output_eval",
                                          per_device_eval_batch_size=32,
                                          do_train=False, do_eval=True,
                                          disable_tqdm=False)
    trainer = transformers.Trainer(model=e2e_late, args=args)

    dataset_iterator = TsvDatasetPredictSep(df, tokenizer)

    trainer.predict(dataset_iterator)


SavedBatchesReader = namedtuple('SavedBatchesReader', ['next_batch', 'close'])


def read_embeddings(embd_type: str, prefix: str):
    handle = open(get_embeddings_path(embd_type, prefix), 'rb')

    max_length = DOC_MAX_LENGTH if embd_type == 'doc' else 24
    is_pad = embd_type == 'query' or "car" not in prefix

    def next_batch():
        try:
            b = [torch.from_numpy(_) for _ in pickle.load(handle)]
            b = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)
            if is_pad:
                b = torch.nn.functional.pad(b, (0, 0, 0, max_length - b.shape[1]))
        except EOFError:
            handle.close()
            return
        return b

    return SavedBatchesReader(next_batch, lambda: handle.close())

# Example run
# generate_embeddings(f"{ARTIFACTS_PATH}/msmarco/sample_collection.csv", "doc", "train")
