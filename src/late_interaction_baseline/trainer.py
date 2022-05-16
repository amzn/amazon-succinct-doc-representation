"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import pandas as pd
import transformers

from src.config import TEMP_OUT_PATH
from src.late_interaction_baseline.data_loader import TsvDatasetSep
from src.late_interaction_baseline.modeling import E2ELate


class MarginMSETrainer(transformers.Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels").reshape((-1, 2))
        output = model(**inputs)
        logits = output.logits.reshape((-1, 2))
        # The diff between pos and neg of the student should be similar to the diff in the teacher.
        # see https://arxiv.org/abs/2010.02666
        return ((logits[:, 0] - logits[:, 1]) - (labels[:, 0] - labels[:, 1]) ** 2).mean()


def shuffle():
    # File was retrieved based on https://arxiv.org/abs/2010.02666
    df = pd.read_csv("trainfile.tsv", sep="\t", header=None)
    df = df.sample(frac=1, random_state=42)
    df.to_csv("trainfile_shuffle.tsv", sep="\t", header=False, index=False)


def train_e2e_late():
    model = transformers.AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    e2e_model = E2ELate(model)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    args = transformers.TrainingArguments(output_dir="~/data/compression", overwrite_output_dir=True,
                                          per_device_train_batch_size=64,
                                          gradient_accumulation_steps=1,
                                          per_device_eval_batch_size=64,
                                          save_steps=10000,
                                          save_total_limit=10,
                                          num_train_epochs=1,
                                          warmup_steps=1000,
                                          max_steps=40000000 // 64,
                                          group_by_length=False,
                                          do_train=True, do_eval=False)
    dfiter = pd.read_csv(f"{TEMP_OUT_PATH}/trainfile_shuffle.tsv", sep="\t", chunksize=10000,
                         names=["pos_score", "neg_score", "query", "pos", "neg"])
    trainer = MarginMSETrainer(model=e2e_model, tokenizer=tokenizer, args=args,
                               train_dataset=TsvDatasetSep(tokenizer, dfiter=dfiter))
    trainer.train()


def main():
    print("Step 1: run process_file from neural-ranking-kd repo. This produces trainfile.tsv")
    shuffle()  # produces trainfile_shuffle.tsv
    train_e2e_late()
    print("Rename latest checkpoint to late_kd_620k_3800 and copy to efs. ")
