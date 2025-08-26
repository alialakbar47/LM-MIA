import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset


def compute_nll(model, token_ids, attention_mask):
    """Computes the mean Negative Log-Likelihood of a batch of sequences."""
    with torch.no_grad():
        outputs = model(token_ids, attention_mask=attention_mask, labels=token_ids)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = token_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss = loss.view(shift_labels.size())

        shift_attention_mask = attention_mask[..., 1:].contiguous()
        masked_loss = loss * shift_attention_mask
        nll = masked_loss.sum(dim=1) / (shift_attention_mask.sum(dim=1) + 1e-9)

        return nll


class SuffixRecallAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.prefix_len = config.get('prefix_len', 50)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        return dataset

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts, return_tensors='pt', padding="longest", truncation=True, max_length=self.tokenizer.model_max_length
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        if token_ids.shape[1] <= self.prefix_len + 1:
            return {self.name: [1.0] * len(texts)}

        suffix_ids = token_ids[:, self.prefix_len:]
        suffix_attention_mask = attention_mask[:, self.prefix_len:]

        # 1. Calculate NLL_unconditional for the suffix
        nll_unconditional = compute_nll(self.model, suffix_ids, suffix_attention_mask)

        # 2. Calculate NLL_conditional for the suffix given the prefix
        with torch.no_grad():
            outputs = self.model(token_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = token_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            suffix_loss_mask = torch.zeros_like(per_token_loss)
            suffix_loss_mask[:, self.prefix_len - 1:] = 1.0

            shift_attention_mask = attention_mask[..., 1:].contiguous()
            final_mask = suffix_loss_mask * shift_attention_mask

            nll_conditional = (per_token_loss * final_mask).sum(dim=1) / (final_mask.sum(dim=1) + 1e-9)

        # 3. Calculate score: nll_unconditional / nll_conditional
        # A lower conditional NLL (more memorized) leads to a higher score.
        scores = nll_unconditional / (nll_conditional + 1e-9)

        return {self.name: scores.cpu().numpy()}