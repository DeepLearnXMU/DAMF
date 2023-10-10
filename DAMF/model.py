from torch.distributions import Categorical
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch

class T5ForConditionalGenerationRAQG(T5ForConditionalGeneration):
    def forward_ra(self, input_ids = None, attention_mask = None, decoder_input_ids = None,
                   labels = None, rewards = None):
        lm_logits = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids).logits # B N V
        _labels = labels.clone()
        _labels[_labels < 0] = 0
        label_matrix = F.one_hot(_labels, self.config.vocab_size).bool()
        logits = F.log_softmax(lm_logits, dim=-1)
        q_prob = logits[label_matrix].reshape_as(labels)
        q_prob[labels < 0] = 0
        q_prob = q_prob.sum(dim=-1)

        _len = (labels > -1).sum(-1)
        len_norm = ((5 + _len) / (5 + 1)) ** 0.5 + 0.5
        _q_prob = q_prob / len_norm # add diversity & length normalize
        _q_prob = F.softmax(_q_prob, dim=-1)

        # _q_prob = F.softmax(q_prob * 0.6, dim=-1) # temperature to add diversity

        sp_indices = Categorical(probs=_q_prob).sample()

        _reward = rewards[sp_indices]

        # bl_indices = torch.argmax(_q_prob, dim=-1)
        # _reward = rewards[sp_indices] - rewards[bl_indices]

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        lm_logits = lm_logits[sp_indices]
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels[sp_indices].view(-1)) # B * N
        loss = loss.mean() * _reward
        return loss

    def teacher_forward(self, **kwargs):
        kwargs.pop('labels')
        model_outputs = self.forward(**kwargs)
        lm_logits = model_outputs.logits
        lm_probs = F.softmax(lm_logits, dim=-1)
        return lm_probs


    def student_forward(self, teacher_probs=None, **kwargs):
        labels = kwargs.pop('labels')
        model_outputs = self.forward(**kwargs)
        lm_logits = model_outputs.logits
        lm_probs = F.softmax(lm_logits, dim=-1)
        decoder_attention_mask = labels != -100
        student_probs = lm_probs[decoder_attention_mask]
        teacher_probs = teacher_probs[decoder_attention_mask]
        loss_fct = torch.nn.CosineEmbeddingLoss()
        target = torch.ones(student_probs.shape[0]).to(student_probs)
        loss = loss_fct(student_probs, teacher_probs, target)
        return loss