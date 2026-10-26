import json
from pathlib import Path

from src.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_fusion_defaults_are_the_dev_selected_config():
    ev = json.loads((ROOT / "results" / "eval.json").read_text())
    decision = ev["dynamic_weights_decision"]
    chosen = ev["configs"][decision["dynamic"] if decision["keep"] else decision["baseline"]]
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    assert defaults["DYNAMIC_WEIGHTS"] is decision["keep"]
    assert defaults["RRF_K"] == chosen["rrf_k"]
    if decision["keep"]:
        assert defaults["DYNAMIC_STRENGTH"] == chosen["strength"]
    else:
        assert (defaults["DENSE_WEIGHT"], defaults["SPARSE_WEIGHT"]) == (1.0, 1.0)
