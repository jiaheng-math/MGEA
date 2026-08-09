from __future__ import annotations

import unittest

from mgea.features import SLOT_FEATURE_NAMES, TRIAGE_FEATURE_NAMES
from mgea.schema import Example
from mgea.slot_allocator import SlotAllocator, SlotConfig
from mgea.triage import DenseProbeTriage, TriageConfig


def make_example(index: int) -> Example:
    dense_sufficient = index < 5
    base_gold = f"base-gold-{index}"
    tail_gold = f"tail-gold-{index}"
    gold = [base_gold] if dense_sufficient else [base_gold, tail_gold]

    dense = []
    for rank in range(20):
        passage_id = base_gold if dense_sufficient and rank == 0 else f"dense-{index}-{rank}"
        dense.append(
            {
                "id": passage_id,
                "title": passage_id,
                "text": f"Dense passage {rank} about Entity {index}",
                "score": 20.0 - rank,
                "source_doc_id": passage_id,
            }
        )

    graph = []
    for rank in range(20):
        if rank == 0:
            passage_id = base_gold
        elif not dense_sufficient and rank == 5:
            passage_id = tail_gold
        else:
            passage_id = f"graph-{index}-{rank}"
        graph.append(
            {
                "id": passage_id,
                "title": passage_id,
                "text": f"Graph passage {rank} about Entity {index}",
                "score": 1.0 - rank * 0.02,
                "source_doc_id": passage_id,
            }
        )

    return Example.from_mapping(
        {
            "id": f"q-{index}",
            "question": f"Which place is linked to Entity {index} and its author?",
            "gold_passage_ids": gold,
            "query_entities": [f"Entity {index}"],
            "retrieval": {"dense": dense, "graph": graph},
        }
    )


class MGEATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = [make_example(index) for index in range(10)]

    def test_feature_dimensions_match_paper_implementation(self) -> None:
        allocator = SlotAllocator()
        record = allocator.candidates(self.examples[5], require_labels=True)[0]
        self.assertEqual(len(TRIAGE_FEATURE_NAMES), 16)
        self.assertEqual(len(SLOT_FEATURE_NAMES), 38)
        self.assertEqual(len(record.features), len(SLOT_FEATURE_NAMES))

    def test_conditional_slot_label(self) -> None:
        allocator = SlotAllocator()
        records = allocator.candidates(self.examples[5], require_labels=True)
        labels = {record.pid: record.label for record in records}
        self.assertEqual(labels["tail-gold-5"], 1)
        self.assertEqual(sum(int(record.label) for record in records), 1)

    def test_triage_labels(self) -> None:
        triage = DenseProbeTriage()
        self.assertEqual(triage.label(self.examples[0]), 0)
        self.assertEqual(triage.label(self.examples[5]), 1)

    def test_grouped_oof_and_average_budget(self) -> None:
        allocator = SlotAllocator(SlotConfig(folds=5, random_seed=7))
        scores, metrics = allocator.out_of_fold(self.examples)
        selected = allocator.allocate(self.examples, scores)
        contexts = allocator.contexts(self.examples, selected)
        self.assertEqual(len(scores), len(self.examples))
        self.assertEqual(len(metrics["fold_by_query"]), len(self.examples))
        self.assertEqual(sum(len(value) for value in selected.values()), 20)
        self.assertTrue(all(len(value) <= 5 for value in selected.values()))
        self.assertAlmostEqual(sum(map(len, contexts.values())) / len(contexts), 7.0)

    def test_triage_oof(self) -> None:
        triage = DenseProbeTriage(TriageConfig(folds=5, random_seed=7))
        scores, metrics = triage.out_of_fold(self.examples)
        decisions = triage.select_invocations(self.examples, scores, target_rate=0.5)
        self.assertEqual(len(scores), len(self.examples))
        self.assertEqual(metrics["dense_sufficient"], 5)
        self.assertEqual(metrics["graph_beneficial"], 5)
        self.assertEqual(sum(decisions.values()), 5)


if __name__ == "__main__":
    unittest.main()
