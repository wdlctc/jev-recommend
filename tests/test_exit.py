import torch

from jevrec.exit import ExitPolicy, ExitScorer, simulate
from test_model import random_batch, scorer, tiny_backbone


def exit_model(layers=4):
    backbone = tiny_backbone("sdpa")
    if layers != 2:
        from transformers import Qwen3Config, Qwen3Model
        torch.manual_seed(0)
        backbone = Qwen3Model(Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                                          num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                                          head_dim=16, max_position_embeddings=256,
                                          attn_implementation="sdpa")).float().eval()
    iso = scorer(backbone, "isolated")
    model = ExitScorer(iso).eval()
    torch.manual_seed(2)
    for h in model.heads[:-1]:
        torch.nn.init.normal_(h.weight, std=0.5)
    return iso, model


def test_last_layer_equals_isolated():
    iso, model = exit_model()
    batch = random_batch(B=5)
    with torch.no_grad():
        torch.testing.assert_close(model(batch)[-1], iso(batch), atol=1e-5, rtol=1e-5)


def test_decide_matches_simulation():
    _, model = exit_model(layers=6)
    batch = random_batch(seed=3, B=16, K=4)
    temps = torch.rand(6) + 0.5
    with torch.no_grad():
        layers = model(batch)
    for policy in [ExitPolicy(), ExitPolicy(tau_hi=0.5), ExitPolicy(patience=2, tau_lo=0.3),
                   ExitPolicy(tau_hi=0.6, patience=3, min_layer=2), ExitPolicy(tau_hi=0.0)]:
        want_logits, want_exit = simulate(layers, temps, policy)
        got_logits, got_exit = model.decide(batch, policy, temps)
        assert torch.equal(got_exit.cpu(), want_exit), policy
        torch.testing.assert_close(got_logits, want_logits, atol=1e-4, rtol=1e-4)
    # No exit rule: everything leaves at the last layer.
    assert (simulate(layers, temps, ExitPolicy())[1] == 5).all()
