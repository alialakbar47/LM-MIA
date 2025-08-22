# FILE: mia_llms_benchmark/attacks/ratio.py (MODIFIED)

from attacks import AbstractAttack
from attacks.utils import batch_nlloss
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class RatioAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.reference_model, self.reference_tokenizer = self._load_reference(config['device'])
        self.reference_device = config['device']

    def _load_reference(self, device):
        reference_model = AutoModelForCausalLM.from_pretrained(self.config['reference_model_path']).to(device)
        reference_tokenizer = AutoTokenizer.from_pretrained(self.config['reference_tokenizer_path'])
        reference_tokenizer.pad_token = reference_tokenizer.eos_token
        return reference_model, reference_tokenizer

    def run(self, dataset: Dataset) -> Dataset:
        # Step 1: Calculate reference NLL. This part is fine.
        dataset = dataset.map(
            lambda x: batch_nlloss(x, self.reference_model, self.reference_tokenizer, self.reference_device, key='reference_nlloss'),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        
        # Step 2: Calculate final score and add column efficiently.
        scores = [-nll / (rnll + 1e-9) for nll, rnll in zip(dataset['nlloss'], dataset['reference_nlloss'])]
        dataset = dataset.add_column(self.name, scores)

        return dataset