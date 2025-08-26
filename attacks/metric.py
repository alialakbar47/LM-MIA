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

        with torch.no_grad():
            outputs = self.model(token_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = token_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            shift_attention_mask = attention_mask[..., 1:].contiguous()

            if per_token_loss.shape[1] < self.prefix_len:
                return {self.name: [100.0] * len(texts)}

            suffix_loss = per_token_loss[:, self.prefix_len - 1:]
            suffix_mask = shift_attention_mask[:, self.prefix_len - 1:]

            suffix_loss_np = suffix_loss.cpu().numpy()
            suffix_mask_np = suffix_mask.cpu().numpy()

            final_scores = []
            for i in range(suffix_loss_np.shape[0]):
                current_losses = suffix_loss_np[i, suffix_mask_np[i] == 1]

                if len(current_losses) == 0:
                    final_scores.append(100.0)
                    continue

                mean = np.mean(current_losses)
                std = np.std(current_losses)
                floor = mean - self.z_threshold * std
                upper = mean + self.z_threshold * std

                metric_loss = np.where(
                    ((current_losses < floor) | (current_losses > upper)),
                    mean,
                    current_losses
                )

                final_scores.append(np.mean(metric_loss))

        return {self.name: np.array(final_scores)}