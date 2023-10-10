# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union, List, Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BatchEncoding, StoppingCriteriaList, LogitsProcessorList, BeamSearchScorer
from transformers.modeling_outputs import (BaseModelOutput,)
from transformers.utils import (ModelOutput)
from transformers.models.t5 import T5ForConditionalGeneration

class RetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class T5ForRAG(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                doc_scores: Optional[torch.FloatTensor] = None,
                context_input_ids: Optional[torch.LongTensor] = None,
                context_attention_mask=None,
                n_docs: Optional[int] = None,
                labels: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                do_marginalize=True,
                ):

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        if encoder_outputs is None:
            retriever_outputs = self.encoder(input_ids, attention_mask, n_docs)

            (
                context_input_ids,
                context_attention_mask,
                context_encoder_outputs,
                doc_scores,
            ) = (
                retriever_outputs["context_input_ids"],
                retriever_outputs["context_attention_mask"],
                retriever_outputs["context_encoder_outputs"],
                retriever_outputs["n_doc_scores"],
            )

            context_encoder_outputs = context_encoder_outputs.reshape((-1,) + context_encoder_outputs.shape[2:])
            context_attention_mask = context_attention_mask.reshape(context_encoder_outputs.shape[:-1])
        else:
            context_encoder_outputs = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        generator_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=context_encoder_outputs,
            encoder_attention_mask=context_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = generator_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.get_nll(
                lm_logits,
                doc_scores,
                labels,
                n_docs,
            )

        if do_marginalize:
            lm_logits = self.marginalize(lm_logits, doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=lm_logits,
            doc_scores=doc_scores,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask, )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            doc_scores=None,
            n_docs=None,
            **kwargs
    ):
        if past is not None:
            # if past is defined use only last decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(hidden_states, new_order):
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result

        reordered_past = ()
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            reordered_past += (tuple(_reorder_stacked(past_state, beam_idx) for past_state in layer_past),)

        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    def get_nll(
            self, seq_logits, doc_scores, target, n_docs
    ):
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        criterion = torch.nn.NLLLoss()
        return criterion(rag_logprobs.view(-1, rag_logprobs.shape[-1]), target.view(-1))

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a RAG model.
        """
        self.encoder = EncoderWrapper(self.encoder)

    def unwrap_encoder(self):
        """
        Unwrap RAG encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder()

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            context_input_ids: Optional[torch.LongTensor] = None,
            context_attention_mask: Optional[torch.LongTensor] = None,
            doc_scores: Optional[torch.FloatTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            early_stopping: Optional[bool] = None,
            use_cache: Optional[bool] = None,
            num_beams: Optional[int] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[List[List[int]]] = None,
            num_return_sequences: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            n_docs: Optional[int] = None,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
            renormalize_logits: Optional[bool] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
            **model_kwargs
    ) -> torch.LongTensor:
        """
        Implements RAG token decoding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether or not to stop the beam search when at least `num_beams` sentences are finished per batch or
                not.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[int]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation_utils.GenerationMixin.generate`] function,
                where we set `num_return_sequences` to `num_beams`. decoder_start_token_id (`int`, *optional*): If an
                encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        exponential_decay_length_penalty = (
            exponential_decay_length_penalty
            if exponential_decay_length_penalty is not None
            else self.config.exponential_decay_length_penalty
        )

        # retrieve docs
        if self.encoder is not None:
            retriever_outputs = self.encoder(input_ids, attention_mask, n_docs)
            (
                context_input_ids,
                context_attention_mask,
                context_encoder_outputs,
                doc_scores,
            ) = (
                retriever_outputs["context_input_ids"],
                retriever_outputs["context_attention_mask"],
                retriever_outputs["context_encoder_outputs"],
                retriever_outputs["n_doc_scores"],
            )
            context_encoder_outputs = context_encoder_outputs.reshape((-1,) + context_encoder_outputs.shape[2:])
            context_attention_mask = context_attention_mask.reshape(context_encoder_outputs.shape[:-1])

            encoder_outputs = BaseModelOutput(last_hidden_state=context_encoder_outputs)

        # batch_size
        batch_size = context_input_ids.shape[0]

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=context_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a RAG model.
    """

    def __init__(self, encoder):
        super().__init__()
        # update to latest transformers
        self.main_input_name = 'input_ids'
        self.encoder = encoder
        self.cls_head = nn.Linear(self.encoder.config.d_model, 1)

    def forward(self, input_ids=None, attention_mask=None, n_docs=None, **kwargs, ):
        # input_ids / attn_mask : B doc_num L
        bsz, doc_num, token_len = input_ids.shape
        _input_ids = input_ids.reshape(bsz * doc_num, token_len)
        _attention_mask = attention_mask.reshape_as(_input_ids)
        outputs = self.encoder(_input_ids, _attention_mask, **kwargs)
        outputs = outputs["last_hidden_state"]
        outputs = outputs.reshape(bsz, doc_num, token_len, -1)
        # cls emb
        cls_emb = outputs[:, :, 0]
        cls_score = self.cls_head(cls_emb).squeeze(-1)  # B doc_num
        n_docs = min(doc_num, n_docs)
        top_values, top_indices = torch.topk(cls_score, k=n_docs, dim=-1, sorted=True)
        # need debug
        context_input_ids = input_ids[torch.arange(bsz), top_indices]
        context_encoder_outputs = outputs[torch.arange(bsz), top_indices]
        context_attention_mask = attention_mask[torch.arange(bsz), top_indices]
        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "context_encoder_outputs": context_encoder_outputs,
                "n_doc_scores": top_values,
                "all_doc_scores": cls_score,
                "encoder_hidden_states": outputs,
            },
            tensor_type='pt',
        )
