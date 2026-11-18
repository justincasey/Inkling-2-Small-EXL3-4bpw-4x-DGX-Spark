import os
import sys

ORACLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ORACLE, os.path.join(ORACLE, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(scope="session")
def tiny():
    from inkling2_ref import DenseExperts, DictWeights, Inkling2Ref
    from synthetic import random_weights, tiny_config

    cfg = tiny_config()
    w = random_weights(cfg, seed=1)
    src = DictWeights(w)

    def make(tp=4, rounding=None):
        return Inkling2Ref(cfg, src, DenseExperts(src), tp=tp, rounding=rounding, unembed_chunk=128, query_chunk=5)

    return cfg, w, make
