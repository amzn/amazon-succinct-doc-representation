"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
import pandas as pd
import torch
import transformers
from scipy.special import softmax
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import TruncationStrategy

from src.config import ARTIFACTS_PATH
from src.evaluation.mrr_metric import eval_mrr
from src.late_interaction_baseline.data_loader import TsvDatasetPredictSep
from src.late_interaction_baseline.modeling import E2ELate


def predict_kd_baseline_scores():
    tokenizer = transformers.BertTokenizerFast.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-12-v2")
    dfiter = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/dev_w_qrels.csv", chunksize=60000)
    args = transformers.TrainingArguments(output_dir="output_eval",
                                          per_device_eval_batch_size=32, do_train=False, do_eval=True)
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=args)
    results = []
    for df in tqdm(dfiter, total=112):
        tokens = tokenizer(df["query"].tolist(), df.doc.tolist(), max_length=256,
                           truncation=TruncationStrategy.LONGEST_FIRST)
        ltokens = [{k: v[i] for k, v in tokens.items()} for i in range(len(tokens["input_ids"]))]
        pred = trainer.predict(ltokens)
        results.append(pred.predictions)
    df = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/dev_w_qrels.csv")
    df["bert_pred_kd"] = np.concatenate([softmax(r, axis=1)[:, 1] for r in results])
    df.to_csv(f"{ARTIFACTS_PATH}/dev_pred_lbls_kd.csv", index=False)


def predict_e2elate_scores(e2e_late: E2ELate = None, df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(f"{ARTIFACTS_PATH}/dev_top10_kd.csv")
    if e2e_late is None:
        e2e_late = E2ELate.from_pretrained(f"{ARTIFACTS_PATH}/msemargin/checkpoint-620000/")
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    df_copy = df.copy(deep=True)
    df["label"] = None
    dataset_iterator = TsvDatasetPredictSep(df, tokenizer, "labels")

    trainer = transformers.Trainer(model=e2e_late, args=transformers.TrainingArguments(
        output_dir="output_eval",
        per_device_eval_batch_size=32,
        do_train=False, do_eval=True,
        disable_tqdm=False,
        seed=torch.seed() % 2 ** 32,
    ))

    preds = trainer.predict(dataset_iterator).predictions.astype(np.float64)

    df = df_copy
    df["e2e_late_prediction"] = preds
    mrr = eval_mrr(df, "e2e_late_prediction")
    print(f'predict_e2elate_scores mrr={mrr}')
    return mrr
