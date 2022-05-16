"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import argparse

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=f"{ARTIFACTS_PATH}/dev_top10_lbls_preds.csv", type=str)
    parser.add_argument("--score_field", default="bert_pred", type=str)
    return parser.parse_args()


def mrr(x: pd.Series, k=10):
    if k is not None:
        x = x[:k]
    return 1 / (x.argmax() + 1) if 1 in x.values else 0


def eval_mrr(df, score_field):
    assert df.label.dtype == np.int64 and df.qid.dtype == np.int64 and df[score_field].dtype in [np.float64, np.float32]
    df = df.sort_values(["qid", score_field], ascending=False)
    mrr_df = df.groupby("qid").label.apply(mrr)
    return mrr_df.mean()


def main(args):
    # ensure that df has: qid, label, and pred(prediction score)
    df = pd.read_csv(args.infile)
    mrr_df = eval_mrr(df, args.score_field)
    print("MRR", mrr_df)


if __name__ == '__main__':
    main(parse_args())
