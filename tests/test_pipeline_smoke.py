from pathlib import Path
from src.data_generator import generate_transactions

def test_sample_fixture_exists_or_generator_is_small():
    assert len(generate_transactions(50,seed=1))==50
    assert Path('data/sample').exists()
