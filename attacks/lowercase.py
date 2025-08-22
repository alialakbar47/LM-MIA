# FILE: mia_llms_benchmark/attacks/lowercase.py (CORRECTED)

from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset


class LowercaseAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        # Step 1: Calculate the NLL of the lowercased text. This part is fine.
        dataset = dataset.map(
            lambda x: self.lowercase_nlloss(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )

        # Step 2: Calculate the final score and add it efficiently.
        scores = [-nll / (lnll + 1e-9) for nll, lnll in zip(dataset['nlloss'], dataset['lowercase_nlloss'])]
        
        # Use add_column to avoid duplicating the dataset in memory.
        dataset = dataset.add_column(self.name, scores)
        
        return dataset

    def lowercase_nlloss(self, batch):
        texts = [x.lower() for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask)
        return {'lowercase_nlloss': losses}