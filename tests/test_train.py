from __future__ import annotations

import torch

from news_classifier.data import Example
from news_classifier.train import PadCollator, TokenizedDataset, tokenize_examples


class FakeTokenizer:
    pad_token_id = 9

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 8
        assert kwargs["padding"] is False
        assert kwargs["return_attention_mask"] is True
        return {"input_ids": [[1, 2], [3]], "attention_mask": [[1, 1], [1]]}


def test_tokenize_examples_builds_map_style_dataset():
    dataset = tokenize_examples(
        [Example("금리 동결", 1), Example("북한 미사일", 4)],
        FakeTokenizer(),
        max_length=8,
    )
    assert isinstance(dataset, TokenizedDataset)
    assert len(dataset) == 2
    assert dataset[0] == {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 1}


def test_pad_collator_pads_to_longest_sequence_in_batch():
    collator = PadCollator(pad_token_id=9)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 0},
            {"input_ids": [3], "attention_mask": [1], "labels": 1},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [0, 1]
    assert batch["input_ids"].dtype is torch.long
