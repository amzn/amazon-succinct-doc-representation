"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import copy
import pickle
import sys
from dataclasses import dataclass
from os.path import join

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel

INITIAL_CROSS_INTERACTION_LAYER = 10


@dataclass
class EmbdsWithOnlyLastLayer:
    last_hidden_state: any


class DocEncoder(BertModel):
    # doc encoder gets tokenized text and output intermediate representation at layer L
    def __init__(self, existing_model: BertModel, config: transformers.BertConfig):
        super().__init__(config)
        self.embeddings = copy.deepcopy(existing_model.embeddings)
        self.encoder = copy.deepcopy(existing_model.encoder)
        self.encoder.layer = nn.ModuleList(list(self.encoder.layer)[:INITIAL_CROSS_INTERACTION_LAYER])
        self.pooler = None


class QueryEncoder(BertModel):
    # query encoder gets tokenized text and output intermediate representation at layer L
    def __init__(self, existing_model: BertModel, config: transformers.BertConfig):
        super().__init__(config)
        self.embeddings = copy.deepcopy(existing_model.embeddings)
        self.encoder = copy.deepcopy(existing_model.encoder)
        self.encoder.layer = nn.ModuleList(list(self.encoder.layer)[:INITIAL_CROSS_INTERACTION_LAYER])
        self.pooler = None


class _IdentityEmbeddings(nn.Module):
    # helper class. Pass intermediate embeddings unmodified.
    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        return inputs_embeds


class QDInteractionModel(BertModel):
    # Given intermediate representation (layer L) from query and doc, run remaining layers (L+1 ... 12)
    def __init__(self, existing_model: BertModel, config: transformers.BertConfig):
        super().__init__(config)
        self.encoder = copy.deepcopy(existing_model.encoder)
        self.encoder.layer = nn.ModuleList(list(self.encoder.layer)[INITIAL_CROSS_INTERACTION_LAYER:])
        self.embeddings = _IdentityEmbeddings()
        self.pooler = existing_model.pooler


class LateInteractionModel(nn.Module):
    # a wrapper module over qd-interaction-model.
    def __init__(self, qd_interaction_model: QDInteractionModel, existing_model: BertForSequenceClassification):
        super(LateInteractionModel, self).__init__()
        if qd_interaction_model is not None:
            self.qd_interaction_model = qd_interaction_model
        else:
            self.qd_interaction_model = QDInteractionModel(existing_model.bert, existing_model.config)
        self.dropout = existing_model.dropout
        self.classifier = existing_model.classifier
        self.num_labels = existing_model.config.num_labels
        self.weight = None

    def set_weight(self, weight=torch.tensor([1.0, 10.0])):
        self.weight = weight.cuda()

    def forward(self,
                query_embeddings=None,
                doc_embeddings=None,
                query_attention_mask=None,
                doc_attention_mask=None,
                labels=None
                # token_type_ids=None,
                ):
        input_embds = torch.cat((query_embeddings.last_hidden_state, doc_embeddings.last_hidden_state), dim=1)
        attention_mask = torch.cat((query_attention_mask, doc_attention_mask), dim=1)
        token_type_ids = torch.cat((torch.zeros_like(query_attention_mask), torch.ones_like(doc_attention_mask)), dim=1)
        outputs = self.qd_interaction_model(inputs_embeds=input_embds, attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        # content here was copied from BertForSequenceClassification.forward.
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if labels.dtype == torch.long or labels.dtype == torch.int:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class E2ELate(BertForSequenceClassification):

    def __init__(self, existing_model: BertForSequenceClassification, query_len=24,
                 save_embds=False,
                 use_saved_embds=False,
                 saved_embds_prefix='eval',
                 doc_embds_compression=None):
        super().__init__(existing_model.config)
        self.doc_encoder = DocEncoder(existing_model.bert, existing_model.config)
        self.query_encoder = QueryEncoder(existing_model.bert, existing_model.config)
        qd_interaction = QDInteractionModel(existing_model.bert, existing_model.config)
        self.late_interaction = LateInteractionModel(qd_interaction, existing_model)
        self.query_len = query_len
        self.enc_dec_sim = None

        # TODO this is very ugly and assumes initialization is synchronized with exact batches
        self.save_embds = save_embds
        self.use_saved_embds = use_saved_embds
        self.doc_embds_compression = doc_embds_compression

        # TODO avoid import cycle with 'precompute_embeddings'
        from src.late_interaction_baseline.precompute_embeddings import get_embeddings_path, read_embeddings

        if save_embds:
            self.doc_embds_file = open(get_embeddings_path('doc', saved_embds_prefix), 'wb')
            self.query_embds_file = open(get_embeddings_path('query', saved_embds_prefix), 'wb')

            self.use_saved_embds = use_saved_embds = False

        if use_saved_embds:
            self.query_batches_reader = read_embeddings("query", saved_embds_prefix)
            self.doc_batches_reader = read_embeddings("doc", saved_embds_prefix)

    def set_enc_dec_sim(self, enc_dec_sim):
        self.enc_dec_sim = enc_dec_sim  # encoder/decoder simulator

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        cutoff = self.query_len
        assert cutoff == (token_type_ids[0, :] == 0).sum(dim=0)

        if self.use_saved_embds:
            query_embds = EmbdsWithOnlyLastLayer(self.query_batches_reader.next_batch().to(input_ids.device))
            doc_embds = EmbdsWithOnlyLastLayer(self.doc_batches_reader.next_batch().to(input_ids.device))
        else:
            query_embds = self.query_encoder(input_ids=input_ids[:, :cutoff],
                                             attention_mask=attention_mask[:, :cutoff],
                                             token_type_ids=token_type_ids[:, :cutoff])
            doc_embds = self.doc_encoder(input_ids[:, cutoff:],
                                         attention_mask[:, cutoff:],
                                         token_type_ids=token_type_ids[:, cutoff:])

        query_attention_mask = attention_mask[:, :cutoff]
        doc_attention_mask = attention_mask[:, cutoff:]

        if self.doc_embds_compression:
            doc_embds.last_hidden_state, doc_attention_mask = self.doc_embds_compression.roundtrip(
                (doc_embds.last_hidden_state, doc_attention_mask))


        if self.save_embds:
            for embds, attention_mask, embds_file in [(query_embds, query_attention_mask, self.query_embds_file),
                                                      (doc_embds, doc_attention_mask, self.doc_embds_file)]:
                embds_last_layer = embds.last_hidden_state.cpu().numpy()
                lengths = attention_mask.sum(dim=1).cpu()
                truncated_embds_last_layer = [v[:l] for v, l in zip(embds_last_layer, lengths)]

                pickle.dump(truncated_embds_last_layer, embds_file)

        if self.enc_dec_sim is not None:
            doc_uncontextualized_embds = self.doc_encoder.embeddings(input_ids=input_ids[:, cutoff:],
                                                                     token_type_ids=token_type_ids[:, cutoff:])
            doc_embds.last_hidden_state = \
                self.enc_dec_sim(doc_embds.last_hidden_state, doc_attention_mask, doc_uncontextualized_embds)

        output = self.late_interaction(query_embeddings=query_embds, doc_embeddings=doc_embds,
                                       query_attention_mask=query_attention_mask,
                                       doc_attention_mask=doc_attention_mask,
                                       labels=labels)
        return output

    @classmethod
    def from_pretrained(cls, path, pretrained_state_dict=None, *model_args, **model_kwargs):
        config = transformers.BertConfig.from_pretrained(join(path, "config.json"))
        base_model = transformers.BertForSequenceClassification(config)
        model = E2ELate(base_model, *model_args, **model_kwargs)
        model.load_state_dict(pretrained_state_dict or torch.load(join(path, "pytorch_model.bin")))
        return model

    def cleanup(self):
        if self.use_saved_embds:
            self.query_batches_reader.close()
            self.doc_batches_reader.close()
        if self.save_embds:
            self.doc_embds_file.close()
            self.query_embds_file.close()


def example():
    model: transformers.BertForSequenceClassification = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-uncased")
    doc_model = DocEncoder(model.bert, model.config)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")
    doc_tokens = tokenizer("hello", return_tensors="pt")
    doc_rep = doc_model(**doc_tokens)

    query_model = QueryEncoder(model.bert, model.config)
    query_tokens = tokenizer("query", return_tensors="pt")
    query_rep = doc_model(**query_tokens)

    # directly apply QD interaction
    qd_model = QDInteractionModel(model.bert, model.config)
    merge = torch.cat((query_rep.last_hidden_state, doc_rep.last_hidden_state), dim=1)
    output_vec = qd_model(inputs_embeds=merge).pooler_output

    late_int = LateInteractionModel(qd_model, model)
    late_int(query_rep[0], doc_rep[0], query_tokens["attention_mask"], doc_tokens["attention_mask"])


print("loaded", sys.modules[__name__])

