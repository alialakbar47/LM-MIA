# FILE: mia_llms_benchmark/attacks/conrecall.py (MODIFIED)

import copy
import random
import datasets
from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset, load_dataset


def make_conrecall_prefix(dataset, n_shots, perplexity_bucket=None, target_index=None):
    prefixes = []
    if target_index is not None:
        indices_to_keep = [i for i in range(len(dataset)) if i != target_index]
        dataset = dataset.select(indices_to_keep)

    if perplexity_bucket is not None:
        datasets.disable_progress_bars()
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
        datasets.enable_progress_bars()

    # Ensure we don't sample more than available examples
    num_available = len(dataset)
    if target_index is not None and target_index in range(num_available):
        all_indices = list(range(num_available))
        all_indices.remove(target_index)
    else:
        all_indices = list(range(num_available))

    n_shots = min(n_shots, len(all_indices))
    indices = random.sample(all_indices, n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)


class ConRecallAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])

    def build_non_member_prefix(self, perplexity_bucket=None):
        return make_conrecall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def build_member_prefix(self, target_index, dataset, perplexity_bucket=None):
        return make_conrecall_prefix(
            dataset=dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket,
            target_index=target_index
        )

    def run(self, dataset: Dataset) -> Dataset:
        # Create a deepcopy for building member prefixes without affecting the main dataset object
        ds_clone = copy.deepcopy(dataset) 
        
        # Step 1: Calculate conditional NLLs. This part is fine.
        dataset = dataset.map(
            lambda x: self.conrecall_nlloss(x, ds_clone),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v7",
        )

        # Step 2: Calculate the final score and add column efficiently.
        nm_nlloss = dataset[f'{self.name}_nm_nlloss']
        m_nlloss = dataset[f'{self.name}_m_nlloss']
        nlloss = dataset['nlloss']
        
        scores = [(nm - m) / (nll + 1e-9) for nm, m, nll in zip(nm_nlloss, m_nlloss, nlloss)]
        dataset = dataset.add_column(self.name, scores)

        return dataset

    def conrecall_nlloss(self, batch, dataset):
        ds_members_only = dataset.filter(lambda x: x["label"] == 1, load_from_cache_file=False)

        if self.config["match_perplexity"]:
            non_member_texts = []
            member_texts = []
            for i, (ppl_bucket, text) in enumerate(zip(batch["perplexity_bucket"], batch["text"])):
                non_member_texts.append(self.build_non_member_prefix(ppl_bucket) + " " + text)
                member_texts.append(self.build_member_prefix(i, ds_members_only, ppl_bucket) + " " + text)
        else:
            non_member_texts = [self.build_non_member_prefix() + " " + text for text in batch["text"]]
            member_texts = [self.build_member_prefix(i, ds_members_only) + " " + text for i, text in enumerate(batch["text"])]

        ret = {}
        for texts, label in [(non_member_texts, "nm"), (member_texts, "m")]:
            tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
            token_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            losses = compute_nlloss(self.model, token_ids, attention_mask)
            ret[f"{self.name}_{label}_nlloss"] = losses
        return ret