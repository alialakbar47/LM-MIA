# FILE: mia_llms_benchmark/attacks/zlib.py (MODIFIED)

import zlib
from attacks import AbstractAttack
from datasets import Dataset


def zlib_score(record):
    text = record["text"]
    loss = record["nlloss"]
    # Add a small epsilon to avoid division by zero for empty strings
    zlib_entropy = len(zlib.compress(text.encode())) / (len(text.encode()) + 1e-9)
    # The score should be higher for members (lower loss).
    zlib_score = -loss / (zlib_entropy + 1e-9)
    return zlib_score

class ZlibAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        # Manually iterate to calculate scores (memory-efficient)
        scores = [zlib_score(record) for record in dataset]
        
        # Use add_column to avoid memory overload
        dataset = dataset.add_column(self.name, scores)
        
        return dataset