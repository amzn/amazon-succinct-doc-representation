"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import itertools
import json
from argparse import ArgumentParser
from datetime import datetime
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.contrib import tzip
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from src.auto_encoder.ae_modeling_training import ContextAutoEncoder
from src.config import ARTIFACTS_PATH, DOC_MAX_LENGTH, TEMP_OUT_PATH
from src.evaluation.mrr_metric import eval_mrr
from src.experiments.utils import print_times, as_batch
from src.late_interaction_baseline.modeling import E2ELate
from src.quantization.drive import Drive


def parse_args():
    parser = ArgumentParser()

    # --ds_csv_path
    #   f'{ARTIFACTS_PATH}/dev_top10_kd.csv'
    #   f'{ARTIFACTS_PATH}/dev_pred_kd_baseline.csv'
    parser.add_argument('--ds_csv_path', type=str, default=f'{ARTIFACTS_PATH}/dev_top25_kd_split.csv')

    parser.add_argument('--ds_chunksize', type=int, default=30000)

    parser.add_argument('--pretrained_model_path', type=str,
                        default=f'{ARTIFACTS_PATH}/msmarco/output/late_kd_620k_3800/')

    parser.add_argument('--ae_modules_checkpoint_path', type=str,
                        default=f'{ARTIFACTS_PATH}/msmarco/output/context_auto_encoder_trial5.plbin')

    parser.add_argument('--do_not_use_saved_embds', dest='use_saved_embds', default=True,
                        action='store_false')  # TODO in python 3.9 we could use, action=BooleanOptionalAction)
    parser.add_argument('--saved_embds_prefix', type=str, default='dev_top25_kd_128')
    parser.add_argument('--save_embds', default=False, action='store_true')

    parser.add_argument('--output_folder', type=str, default="partial")

    parser.add_argument('--compression_seed_start', type=int, default=42)
    parser.add_argument('--compression_bits', type=int, default=4)

    return parser.parse_args()


class ChunkedPredictData(torch.utils.data.IterableDataset):
    def __init__(self, df_iter, tokenizer: transformers.PreTrainedTokenizer):

        self.df_iter = df_iter
        self.tokenizer = tokenizer

    @print_times
    def tokenize(self, df):
        queries = df['query'].tolist()
        passages = df['doc'].tolist()

        q_tokens = self.tokenizer(queries, max_length=24, truncation=TruncationStrategy.LONGEST_FIRST,
                                  padding=PaddingStrategy.MAX_LENGTH, return_tensors='pt')
        d_tokens = self.tokenizer(passages, max_length=DOC_MAX_LENGTH, truncation=TruncationStrategy.LONGEST_FIRST,
                                  padding=PaddingStrategy.MAX_LENGTH, return_tensors='pt')
        # document is the second piece of text, even though it is encoded as a single text. Therefore, type_id is 1
        d_tokens['token_type_ids'][:, :] = 1

        return [torch.cat((q_tokens[k], d_tokens[k]), dim=1) for k in q_tokens.keys()]

    def __iter__(self):
        for chunk_idx, df in enumerate(self.df_iter, start=1):
            for input_ids, token_type_ids, attention_mask in tzip(*self.tokenize(df),
                                                                  desc=f'process chunk {chunk_idx}'):
                yield {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                }


@print_times
def df_iter_to_dataset(df_iter):
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    return ChunkedPredictData(df_iter, tokenizer)


@print_times
def read_data_csv(csv_path, chunksize=None, usecols=None):
    return pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols)


@print_times
def load_model_state_dict(pretrained_model_path):
    return torch.load(join(pretrained_model_path, 'pytorch_model.bin'))


@print_times
def load_ae_modules(ae_modules_checkpoint_path):
    return ContextAutoEncoder.load_from_checkpoint(ae_modules_checkpoint_path).enc_dec_modules.cuda()


@print_times
def load_trainer(pretrained_model_path, model_state_dict, *model_args, **model_kwargs):
    e2e_late = E2ELate.from_pretrained(pretrained_model_path, model_state_dict, *model_args, **model_kwargs)

    trainer = transformers.Trainer(model=e2e_late, args=transformers.TrainingArguments(
        output_dir='output_eval',
        per_device_eval_batch_size=128,
        do_train=False, do_eval=True,
        disable_tqdm=False,
        seed=torch.seed() % 2 ** 32,
    ))

    return trainer


@print_times
def predict(trainer, dataset):
    return trainer.predict(dataset).predictions.squeeze()


def get_seed_generator(seed):
    gen = np.random.default_rng(seed=seed)

    def gen_seed():
        return int(gen.integers(2 ** 63, dtype=np.int64))

    return gen_seed


# The following constants may be edited manually before each run
# specifically these list determines a test grid of
# autoencoder types, size of autoencoder dimensions, and compression types
AE_MODULES_NAMES = ['AESI', 'AutoEncoderWithDecoderSideInfo', 'OneLayerAutoEncoder', 'OneLayerAutoEncoderWithUncontext', 'AutoEncoder2Layers'] #['NoAE']
AE_NUM_FEATURES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
COMPRESSIONS = [
    # ('NOOP', lambda seed, bits, ae_num_features: NoopTransform.s), #0.3756068358575522

    # ('DR', lambda seed, bits, ae_num_features: as_batch(
    #      ae_num_features, DeterministicRounding(bits=bits, batched=True)
    # )),
    # ('SR', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, StochasticQuantization(bits=bits, subtractive=False, batched=True)
    # )),
    # ('SD', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, StochasticQuantization(bits=bits, subtractive=True, batched=True)
    # )),
    # ('H_DR', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, HadamardDR(gen_seed=get_seed_generator(seed), bits=bits, batched=True)
    # )),
    # ('H_SR', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, HadamardSQ(gen_seed=get_seed_generator(seed), bits=bits, subtractive=False, batched=True)
    # )),
    # ('H_SD', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, HadamardSQ(gen_seed=get_seed_generator(seed), bits=bits, subtractive=True, batched=True)
    # )),
    # ('DRIVE', lambda seed, bits, ae_num_features: as_batch(
    #     ae_num_features, Drive(gen_seed=get_seed_generator(seed), bits=bits, bias_correction=True, batched=True)
    # )),
    ('DRIVE_NBC', lambda seed, bits, ae_num_features: as_batch(
        ae_num_features, Drive(gen_seed=get_seed_generator(seed), bits=bits, bias_correction=False, batched=True)
    )),

    # ('LOOP_DRIVE_6b', lambda seed, ae_num_features: EmbdsBatchLoop(
    #     max_tokens_per_embd=DOC_MAX_LENGTH,
    #     inner_transform=Drive(gen_seed=get_seed_generator(seed), bits=6)
    # )),
    # ('DRIVE_6b_r10', RepeatTransform(Drive(bits=6), 10)),
]

print('\n\n!!! Running the following Experiments:')
print('\n'.join([f'{ae_module_name}-{ae_num_features}, {compression_name}' for
                 ae_module_name, ae_num_features, (compression_name, compression) in
                 itertools.product([_ for _ in AE_MODULES_NAMES if _ != 'NoAE'],
                                   AE_NUM_FEATURES,
                                   COMPRESSIONS)]))
if 'NoAE' in AE_MODULES_NAMES:
    print('\n'.join([f'NoAE-384, {compression_name}' for compression_name, compression in COMPRESSIONS]))
print('\n\n')


def run_experiment():
    args = parse_args()

    full_df = read_data_csv(args.ds_csv_path, chunksize=None, usecols=['qid', 'label'])
    ae_modules = load_ae_modules(args.ae_modules_checkpoint_path)

    Path(f'{TEMP_OUT_PATH}/{args.output_folder}').mkdir(parents=True, exist_ok=True)
    for compression_name, compression_factory in COMPRESSIONS:
        compression_seed_start = args.compression_seed_start
        for ae_module_name in AE_MODULES_NAMES:
            for ae_num_features in (AE_NUM_FEATURES if ae_module_name != 'NoAE' else [384]):
                print(
                    f"start experiment (net={ae_module_name}-{ae_num_features}; compression={compression_name}-{args.compression_bits})")

                df_iter = read_data_csv(args.ds_csv_path, args.ds_chunksize)
                dataset = df_iter_to_dataset(df_iter)

                compression = compression_factory(compression_seed_start, args.compression_bits, ae_num_features)
                compression_seed_start += 1

                model_state_dict = load_model_state_dict(args.pretrained_model_path)
                trainer = load_trainer(args.pretrained_model_path, model_state_dict,
                                       save_embds=args.save_embds,
                                       use_saved_embds=args.use_saved_embds,
                                       saved_embds_prefix=args.saved_embds_prefix)

                if ae_module_name != 'NoAE':
                    ae_module = ae_modules[f'{ae_module_name}_{ae_num_features}']

                    def enc_dec_simulation(doc_vecs, attention_mask, uncontext_doc_vecs):
                        encoded = ae_module.encode(doc_vecs, uncontext_doc_vecs)

                        encoded, attention_mask = compression.roundtrip((encoded, attention_mask))

                        return ae_module.decode(encoded, uncontext_doc_vecs)

                    trainer.model.set_enc_dec_sim(enc_dec_simulation)
                else:
                    trainer.model.doc_embds_compression = compression

                preds = predict(trainer, dataset)

                trainer.model.cleanup()

                np.save(
                    f'{TEMP_OUT_PATH}/{args.output_folder}/preds_{ae_module_name}_{ae_num_features}_{compression_name}_{args.compression_bits}',
                    preds)

                full_df['pred'] = preds

                mrr = eval_mrr(full_df, 'pred')

                print(
                    f"{mrr=} (net={ae_module_name}-{ae_num_features}; compression={compression_name}-{args.compression_bits})")
                with open(f"{TEMP_OUT_PATH}/{args.output_folder}/results.json", "a+") as handle:
                    json.dump({
                        "ae_module": ae_module_name,
                        'num_features': ae_num_features,
                        "compression": compression_name,
                        "bits_per_feature": args.compression_bits,
                        "mrr": mrr,
                        "time": str(datetime.now())}, handle)
                    handle.write("\n")


if __name__ == '__main__':
    run_experiment()
else:
    raise RuntimeError('this file is only meant to run as __main__')
