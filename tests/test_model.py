import random

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from jevrec.model import DecisionScorer, block_mask, pack_setwise, STATE, DECIDE, PAD


def tiny_backbone(attn):
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=256, attn_implementation=attn)
    return Qwen3Model(config).float().eval()


def random_batch(seed=0, B=3, K=4):
    rng = random.Random(seed)
    tok = lambda n: [rng.randrange(1, 97) for _ in range(n)]  # noqa: E731
    return [{"state": tok(rng.randint(5, 12)), "cands": [tok(rng.randint(2, 6)) for _ in range(K)],
             "decide": tok(3), "label": 0} for _ in range(B)]


def scorer(backbone, mode):
    torch.manual_seed(1)
    return DecisionScorer(backbone, 64, torch.randn(64), mode=mode, pad_id=0).eval()


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_isolated_equals_pointwise(attn):
    backbone = tiny_backbone(attn)
    batch = random_batch()
    with torch.no_grad():
        point = scorer(backbone, "pointwise")(batch)
        iso = scorer(backbone, "isolated")(batch)
    torch.testing.assert_close(iso, point, atol=1e-5, rtol=1e-5)


def test_listwise_starts_equal_to_isolated_and_uses_other_candidates():
    backbone = tiny_backbone("sdpa")
    batch = random_batch()
    iso, lst = scorer(backbone, "isolated"), scorer(backbone, "listwise")
    with torch.no_grad():
        torch.testing.assert_close(lst(batch), iso(batch))
        torch.nn.init.normal_(lst.query.weight)
        base = lst(batch)
        changed = [dict(r, cands=[r["cands"][0], r["cands"][1], [5, 6, 7], r["cands"][3]]) for r in batch]
        after = lst(changed)
    # Swapping candidate 2 moves candidate 0's score only through the decide segment.
    assert not torch.allclose(base[:, 0], after[:, 0])
    with torch.no_grad():
        torch.testing.assert_close(iso(changed)[:, 0], iso(batch)[:, 0])


def test_isolated_scores_do_not_depend_on_other_candidates():
    backbone = tiny_backbone("sdpa")
    model = scorer(backbone, "isolated")
    batch = random_batch(K=3)
    with torch.no_grad():
        full = model(batch)
        reversed_batch = [dict(r, cands=r["cands"][::-1]) for r in batch]
        rev = model(reversed_batch)
    torch.testing.assert_close(rev, full.flip(-1), atol=1e-5, rtol=1e-5)


def test_block_mask_structure():
    ids, pos, seg, cand_last, decide_last = pack_setwise(
        [{"state": [1, 2], "cands": [[3], [4, 5]], "decide": [6]}], True, pad_id=0)
    assert seg.tolist() == [[STATE, STATE, 1, 2, 2, DECIDE]]
    assert pos.tolist() == [[0, 1, 2, 2, 3, 4]]
    assert cand_last.tolist() == [[2, 4]] and decide_last.tolist() == [5]
    allowed = block_mask(seg, torch.float32)[0, 0] == 0
    assert allowed[3].tolist() == [True, True, False, True, False, False]   # cand 2 skips cand 1
    assert allowed[5].all()                                                 # decide sees all
    pads = pack_setwise([{"state": [1], "cands": [[2]], "decide": [3]},
                         {"state": [1, 2, 3], "cands": [[2]], "decide": [3]}], False, 0)[2]
    assert pads[0].tolist()[-2:] == [PAD, PAD]
