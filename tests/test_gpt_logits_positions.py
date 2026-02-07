import torch

from nanochat.gpt import GPT, GPTConfig


def test_gpt_forward_logits_positions_shapes():
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, config.vocab_size, (3, 7), dtype=torch.long)

    full = model(idx)
    assert full.shape == (3, 7, config.vocab_size)

    pos = torch.tensor([0, 3, 6], dtype=torch.long)
    sliced = model(idx, logits_positions=pos)
    assert sliced.shape == (3, config.vocab_size)

    last = model(idx, logits_positions=idx.size(1) - 1)
    assert last.shape == (3, config.vocab_size)

