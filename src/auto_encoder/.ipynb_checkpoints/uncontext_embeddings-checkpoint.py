import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import pytorch_lightning as pl
from torch import nn
import torch

from src.auto_encoder.process_doc_vecs import read_doc_embeddings
from src.config import ARTIFACTS_PATH, DOC_MAX_LENGTH
from src.evaluation.evaluate import eval_mrr
from src.naive_late_interaction.data_loader import TsvDatasetPredictSep
from src.naive_late_interaction.modeling import E2ELate
from scipy.special import softmax

# TODO: make this configurable.
TRIALS_NUM_FEATURES = [1, 2, 3, 6] + list(range(4, 64 + 1, 4))
TRIAL_NUM = 5


# TRIALS_NUM_FEATURES = list(range(4, 64 + 1, 4))
# TRIAL_NUM = 3


# TRIALS_NUM_FEATURES = list(range(1, 4 + 1, 1))
# TRIAL_NUM = 4

class AbstractAutoEncoder(nn.Module, ABC):

    @abstractmethod
    def encode(self, cont_vecs, uncont_vecs):
        pass

    @abstractmethod
    def decode(self, encoded, uncont_vecs):
        pass

    def forward(self, cont_vecs, uncont_vecs):
        return self.decode(self.encode(cont_vecs, uncont_vecs), uncont_vecs)


class AutoEncoderCNN(AbstractAutoEncoder):
    def __init__(self, hidden_size=384, inner_size=768 * 2, enc_size=16):
        super().__init__()
        self.e1 = torch.nn.Conv1d(hidden_size * 2, inner_size, 5, padding=2)
        self.e2 = torch.nn.Conv1d(inner_size, enc_size, 5, padding=2)
        self.d1 = torch.nn.Conv1d(enc_size + hidden_size, inner_size, 5, padding=2)
        self.d2 = torch.nn.Conv1d(inner_size, hidden_size, 5, padding=2)
        self.activation = torch.nn.GELU()
        self.hidden_size = hidden_size

    # This class expects a different format of input:
    # Instead of [bs, seq_len, hidden_size], it expects [bs, hidden_size, seq_len]
    def encode(self, cont_vecs, uncont_vecs):
        assert cont_vecs.size(1) == self.hidden_size
        assert uncont_vecs.size(1) == self.hidden_size
        merged_vecs = torch.cat((cont_vecs, uncont_vecs), dim=1)
        encoded = self.e2(self.activation(self.e1(merged_vecs)))
        return encoded

    def decode(self, encoded, uncont_vecs):
        encoded_with_uncont = torch.cat((encoded, uncont_vecs), dim=1)
        reconstructed = self.d2(self.activation(self.d1(encoded_with_uncont)))
        return reconstructed


class AutoEncoderWithSideInfoExp(AbstractAutoEncoder):
    def __init__(self, hidden_size=384, inner_size=768 * 2, enc_size=128):
        super().__init__()
        self.e1 = nn.Linear(hidden_size * 2, inner_size)
        self.e2 = nn.Linear(inner_size, enc_size)
        self.d1 = nn.Linear(enc_size + hidden_size, inner_size)
        self.d2 = nn.Linear(inner_size, hidden_size)
        self.activation = torch.nn.GELU()

    # full encoding with context
    def encode(self, cont_vecs, uncont_vecs):
        merged_vecs = torch.cat((cont_vecs, uncont_vecs), dim=2)
        encoded = self.e2(self.activation(self.e1(merged_vecs)))
        return encoded

    def decode(self, encoded, uncont_vecs):
        encoded_with_uncont = torch.cat((encoded, uncont_vecs), dim=2)
        reconstructed = self.d2(self.activation(self.d1(encoded_with_uncont)))
        return reconstructed


class AutoEncoderWithDecoderSideInfo(AbstractAutoEncoder):
    # during encode, don't provide uncontext.
    def __init__(self, hidden_size=384, inner_size=768 * 2, enc_size=128):
        super().__init__()
        self.e1 = nn.Linear(hidden_size, enc_size)
        self.d1 = nn.Linear(enc_size + hidden_size, inner_size)
        self.d2 = nn.Linear(inner_size, hidden_size)
        self.activation = torch.nn.GELU()

    def encode(self, cont_vecs, uncont_vecs):
        encoded = self.e1(cont_vecs)
        return encoded

    def decode(self, encoded, uncont_vecs):
        encoded_with_uncont = torch.cat((encoded, uncont_vecs), dim=2)
        reconstructed = self.d2(self.activation(self.d1(encoded_with_uncont)))
        return reconstructed


class OneLayerAutoEncoder(AbstractAutoEncoder):
    # simple auto encoder baseline.
    def __init__(self, hidden_size=384, inner_size=None, enc_size=128):
        super().__init__()
        assert inner_size is None
        self.e1 = nn.Linear(hidden_size, enc_size)
        self.d1 = nn.Linear(enc_size, hidden_size)

    def encode(self, cont_vecs, uncont_vecs):
        encoded = self.e1(cont_vecs)
        return encoded

    def decode(self, encoded, uncont_vecs):
        reconstructed = self.d1(encoded)
        return reconstructed


class OneLayerAutoEncoderWithUncontext(AbstractAutoEncoder):
    # simple auto encoder with uncontext. No activation.
    def __init__(self, hidden_size=384, inner_size=None, enc_size=128):
        super().__init__()
        assert inner_size is None
        self.e1 = nn.Linear(hidden_size, enc_size)
        self.d1 = nn.Linear(enc_size + hidden_size, hidden_size)

    def encode(self, cont_vecs, uncont_vecs):
        encoded = self.e1(cont_vecs)
        return encoded

    def decode(self, encoded, uncont_vecs):
        encoded_with_uncontext = torch.cat((encoded, uncont_vecs), dim=2)
        reconstructed = self.d1(encoded_with_uncontext)
        return reconstructed


class AutoEncoder2Layers(AbstractAutoEncoder):
    # auto encoder 2 layers. TODO: fix inner size to 768*2
    def __init__(self, hidden_size=384, inner_size=256, enc_size=128):
        super().__init__()
        self.e1 = nn.Linear(hidden_size, inner_size)
        self.e2 = nn.Linear(inner_size, enc_size)
        self.d1 = nn.Linear(enc_size, inner_size)
        self.d2 = nn.Linear(inner_size, hidden_size)
        self.activation = torch.nn.GELU()

    def encode(self, cont_vecs, uncont_vecs):
        encoded = self.e2(self.activation(self.e1(cont_vecs)))
        return encoded

    def decode(self, encoded, uncont_vecs):
        reconstructed = self.d2(self.activation(self.d1(encoded)))
        return reconstructed


class ContextAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_size=384):
        super().__init__()
        modules = dict()
        for num_features in TRIALS_NUM_FEATURES:
            for mod in [AutoEncoderWithSideInfoExp, AutoEncoderWithDecoderSideInfo, OneLayerAutoEncoder,
                        OneLayerAutoEncoderWithUncontext, AutoEncoder2Layers]:
                modules[f"{mod.__name__}_{num_features}"] = mod(enc_size=num_features, hidden_size=hidden_size)
        self.enc_dec_modules = torch.nn.ModuleDict(modules)

    def forward(self, batch):  # Not implemented
        return batch

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        cont_vecs = batch["cont_vecs"]  # shape (BS=1, SeqLen, hidden_size=384)
        uncont_vecs = batch["uncont_vecs"]  # shape (BS=1, SeqLen, hidden_size=384)
        loss_func = nn.MSELoss()
        losses = [loss_func(module(cont_vecs, uncont_vecs), cont_vecs)
                  for name, module in self.enc_dec_modules.items()]
        if batch_idx % 1000 == 0:
            self.log("losses", {name: round(ls.item(), 3) for ls, name in zip(losses, self.enc_dec_modules.keys())},
                     prog_bar=True)
        return torch.stack(losses).sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=1)
        return [optimizer], [scheduler]


class ContextVectorDataset(torch.utils.data.IterableDataset):
    def __init__(self, embeddings, tokenizer):
        self.embeddings = embeddings
        self.tokenizer = tokenizer

    def __iter__(self):
        for entry in read_doc_embeddings("/home/ubuntu/data/compression/doc_embeddings_kd620k_1m.pkl"):
            vectors = [x[0] for x in entry]
            passages = [x[2] for x in entry]
            d_tokens = self.tokenizer(passages, max_length=DOC_MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST,
                                      padding=PaddingStrategy.MAX_LENGTH, return_tensors="pt")
            d_tokens["token_type_ids"][:, :] = 1
            uncontext_embds = self.embeddings(input_ids=d_tokens["input_ids"],
                                              token_type_ids=d_tokens["token_type_ids"])
            uncontext_embds = uncontext_embds.detach()
            for cont_embds, uncont_embds in zip(vectors, uncontext_embds):
                yield {"cont_vecs": torch.tensor(cont_embds).T,
                       "uncont_vecs": uncont_embds[:cont_embds.shape[0], :].T}


def train():
    e2e_late: E2ELate = E2ELate.from_pretrained(f"{ARTIFACTS_PATH}/msmarcos/output/late_kd_620k_3800/")
    embeddings = e2e_late.doc_encoder.embeddings
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    aemodel = ContextAutoEncoder(e2e_late.doc_encoder.config.hidden_size)
    # instead of using padding like normal people, we don't pad, then accumulate gradients.
    # This is likely slower, but who cares.
    dl = torch.utils.data.DataLoader(dataset=ContextVectorDataset(embeddings, tokenizer), batch_size=1, num_workers=1)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=256, max_epochs=1)
    trainer.fit(aemodel, dl)
    trainer.save_checkpoint(f"{ARTIFACTS_PATH}/msmarcos/output/context_auto_encoder_trial{TRIAL_NUM}.plbin")


def evaluate():
    # def eval():
    enc_dec_model = ContextAutoEncoder.load_from_checkpoint(
        f"{ARTIFACTS_PATH}/msmarcos/output/context_auto_encoder_trial{TRIAL_NUM}.plbin").cuda()
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    args = transformers.TrainingArguments(output_dir="output_eval",
                                          per_device_eval_batch_size=32, do_train=False, do_eval=True,
                                          disable_tqdm=False)
    for name, module in enc_dec_model.enc_dec_modules.items():
        def enc_dec_simulation(doc_vecs, attention_mask, uncontext_doc_vecs):
            return module(doc_vecs, uncontext_doc_vecs)

        e2e_late: E2ELate = E2ELate.from_pretrained(f"{ARTIFACTS_PATH}/msmarcos/output/late_kd_620k_3800/")
        # , use_saved_embds=True) # TODO: restore when we have them.
        trainer = transformers.Trainer(model=e2e_late, args=args)
        e2e_late.set_enc_dec_sim(enc_dec_simulation)
        df = pd.read_csv(f"{ARTIFACTS_PATH}/dev_top10_kd.csv")
        df_copy = df.copy(deep=True)
        df["label"] = None
        preds = trainer.predict(TsvDatasetPredictSep(df, tokenizer))
        df = df_copy
        df[f"prediction_{name}"] = preds.predictions.astype(np.float64)
        # df[f"prediction_{name}"] = softmax(preds.predictions, axis=1)[:, 1]
        # df[f"prediction_{name}"] = df[f"prediction_{name}"].astype(np.float64)
        mrr = eval_mrr(df, f"prediction_{name}")
        print(f"{name=}. {mrr=}")
        with open(f"{ARTIFACTS_PATH}/results.json", "a+") as handle:
            json.dump({"name": name, "mrr": mrr, "time": str(datetime.now())}, handle)
            handle.write("\n")
        # df.to_csv("artifacts/dev_top10_ae_with_uncontext_res.csv", index=False)


if __name__ == '__main__':
    # train()
    evaluate()

print(f"Loaded {sys.modules[__name__]}")
