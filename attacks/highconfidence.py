import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset


class HighConfidenceAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.prefix_len = config.get('prefix_len', 50)
        self.threshold = config.get('threshold', 0.5)
        self.factor = config.get('factor', 0.15)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v2_corrected",
        )
        return dataset

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts, return_tensors='pt', padding="longest", truncation=True, max_length=self.tokenizer.model_max_length
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(token_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = token_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss_flat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            top_scores, _ = shift_logits.topk(2, dim=-1)
            flag1 = (top_scores[..., 0] - top_scores[..., 1]) > self.threshold
            flag2 = top_scores[..., 0] > 0

            flat_flag1 = flag1.view(-1)
            flat_flag2 = flag2.view(-1)

            shift_attention_mask = attention_mask[..., 1:].contiguous()
            mean_batch_loss = (per_token_loss_flat * shift_attention_mask.view(-1)).sum() / shift_attention_mask.sum()

            loss_adjusted_flat = per_token_loss_flat - (flat_flag1.int() - flat_flag2.int()) * mean_batch_loss * self.factor
            loss_adjusted_reshaped = loss_adjusted_flat.view(shift_labels.size())

            if loss_adjusted_reshaped.shape[1] < self.prefix_len:
                # Assign a high loss (low score after negation) for sequences too short
                scores = torch.full((len(texts),), 100.0, device=self.device)
            else:
                loss_adjusted_suffix = loss_adjusted_reshaped[:, self.prefix_len - 1:]
                suffix_mask = shift_attention_mask[:, self.prefix_len - 1:]
                scores = (loss_adjusted_suffix * suffix_mask).sum(dim=1) / (suffix_mask.sum(dim=1) + 1e-9)
        
        # CRITICAL FIX: We negate the score.
        # This attack calculates a modified loss, where a LOWER value means more likely to be a member.
        # The evaluation framework (AUC) expects a HIGHER score for members.
        # Negating the loss achieves this.
        return {self.name: -scores.cpu().numpy()}