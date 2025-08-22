# FILE: mia_llms_benchmark/attacks/loss.py (CORRECTED)

from attacks import AbstractAttack
from datasets import Dataset

class LossAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        """
        Calculates the loss score.
        The score is simply the negative of the pre-computed 'nlloss'.
        """
        # Manually compute the scores in a simple list comprehension.
        scores = [-x for x in dataset['nlloss']]
        
        # Use add_column to add the new scores. This is highly memory-efficient.
        dataset = dataset.add_column(self.name, scores)
        
        return dataset