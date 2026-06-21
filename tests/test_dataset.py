import pytest
import tempfile
import os
from pathlib import Path

from evals.core.dataset import EvalDataset
from evals.core.interfaces import EvalCase


def test_dataset_loading_and_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy jsonl files
        p1 = Path(tmpdir) / "unit"
        p1.mkdir()
        with open(p1 / "test1.jsonl", "w") as f:
            f.write('{"id": "1", "input": "test", "tags": ["t1"]}\n')
            f.write('{"id": "2", "input": "test2", "tags": ["t1", "t2"], "difficulty": "hard"}\n')
            f.write('invalid json line\n')
            
        p2 = Path(tmpdir) / "e2e"
        p2.mkdir()
        with open(p2 / "test2.jsonl", "w") as f:
            f.write('{"id": "3", "input": "test3", "tags": ["t3"]}\n')
            
        ds = EvalDataset(tmpdir)
        
        # Test loading (skips invalid)
        assert len(ds) == 3
        
        # Test implicit category assignment
        assert ds.get_case("1").category == "unit"
        assert ds.get_case("3").category == "e2e"
        
        # Test filtering
        t1_ds = ds.filter_by_tags(["t1"])
        assert len(t1_ds) == 2
        
        cat_ds = ds.filter_by_category("e2e")
        assert len(cat_ds) == 1
        
        diff_ds = ds.filter_by_difficulty("hard")
        assert len(diff_ds) == 1
        
        # Test sampling
        sample_ds = ds.sample(2)
        assert len(sample_ds) == 2
        
        # Test summary
        summary = ds.summary()
        assert summary["total_cases"] == 3
        assert summary["by_tag"]["t1"] == 2
        assert summary["by_category"]["unit"] == 2
