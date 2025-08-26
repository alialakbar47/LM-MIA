import torch
import torch.nn.functional as F
import numpy as np
from attacks import AbstractAttack
from datasets import Dataset


class MetricAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.prefix_len = config.get('prefix_len', 50)
        self.z_threshold = config.get('z_threshold', 3.0)

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
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            if per_token_loss.shape[1] < self.prefix_len:
                # Assign a high loss (low score after negation) for sequences too short
                scores = np.full((len(texts),), -100.0)
                return {self.name: scores}

            suffix_loss = per_token_loss[:, self.prefix_len - 1:]
            suffix_mask = shift_attention_mask[:, self.prefix_len - 1:]

            # Vectorized implementation for efficiency and correctness
            suffix_loss_np = suffix_loss.cpu().numpy()
            suffix_mask_np = suffix_mask.cpu().numpy().astype(bool)

            # Set masked values to NaN to ignore them in mean/std calculations
            suffix_loss_np[~suffix_mask_np] = np.nan
            
            with np.errstate(invalid='ignore'): # Ignore warnings for rows that are all NaNs
                mean = np.nanmean(suffix_loss_np, axis=1, keepdims=True)
                std = np.nanstd(suffix_loss_np, axis=1, keepdims=True)
            
            floor = mean - self.z_threshold * std
            upper = mean + self.z_threshold * std

            # Replace outliers with the mean of their own sequence
            metric_loss = np.where(
                ((suffix_loss_np < floor) | (suffix_loss_np > upper)),
                mean,
                suffix_loss_np
            )

            with np.errstate(invalid='ignore'):
                scores = np.nanmean(metric_loss, axis=1)

            # Handle cases where a sequence had no valid suffix tokens, resulting in NaN
            # Assign a high loss (low score after negation) to these cases.
            scores = np.nan_to_num(scores, nan=100.0)

        # CRITICAL FIX: We negate the score.
        # This attack calculates an outlier-adjusted loss. A LOWER loss indicates membership.
        # We negate it so a HIGHER score indicates membership for the AUC calculation.
        return {self.name: -scores}