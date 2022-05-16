"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import pandas as pd
import transformers
from transformers.tokenization_utils_base import TruncationStrategy
from scipy.special import softmax
from tqdm.auto import tqdm
import numpy as np

from src.config import ARTIFACTS_PATH


def merge_data_with_qrels():
    df = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/top1000.dev", sep="\t", names=["qid", "docid", "query", "doc"])
    dfqrels = pd.read_csv(f"{ARTIFACTS_PATH}/msmarco/qrels.dev.tsv", sep="\t", names=["qid", 0, "docid", "label"])
    del dfqrels[0]
    dfm = df.merge(dfqrels, on=["qid", "docid"], how="left")
    dfm.label = dfm.label.fillna(0).astype(int)
    dfm.to_csv(f"{ARTIFACTS_PATH}/msmarco/dev_w_qrels.csv", index=False)


def predict_scores():
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        f"{ARTIFACTS_PATH}/msemargin/checkpoint-620000/")
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
    df["bert_pred"] = np.concatenate([softmax(r, axis=1)[:, 1] for r in results])
    df.to_csv(f"{ARTIFACTS_PATH}/dev_pred_lbls.csv", index=False)


def select_top_k(k=50):
    df = pd.read_csv(f"{ARTIFACTS_PATH}/dev_pred_lbls.csv")
    df.sort_values("bert_pred", ascending=False, inplace=True)
    df2 = df.groupby("qid").head(k)
    # Next, if a query does not have any positive, keep only one sample.
    # Model will anyway fails, so no need to predict every negative sample.
    has_positive_label_df = df2.groupby("qid").label.any()
    qids_with_pos_doc = set(has_positive_label_df[has_positive_label_df].reset_index()["qid"])
    df_pos = df2[df2.qid.isin(qids_with_pos_doc)]
    df_no_pos = df2[~ df2.qid.isin(qids_with_pos_doc)]
    df_no_pos = df_no_pos.groupby("qid").head(1)
    df = pd.concat((df_pos, df_no_pos)).sort_values(["qid", "bert_pred"], ascending=False)
    df.to_csv(f"{ARTIFACTS_PATH}/dev_top{k}.csv", index=False)


merge_data_with_qrels()
predict_scores()
select_top_k(k=50)
