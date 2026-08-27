import pytest
import torch
from freetoken.engine.draft_runner import rejection_sample, _sampling_probs
from freetoken.server.args import parse_args


def test_rejection_sample_greedy_full_accept():
    gamma = 4
    vocab_size = 100
    draft_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    
    # Target logits strongly match draft tokens for all gamma positions
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 10] = 1.0
    target_probs[0, 1, 20] = 1.0
    target_probs[0, 2, 30] = 1.0
    target_probs[0, 3, 40] = 1.0
    target_probs[0, 4, 50] = 1.0  # Next token after full acceptance
    
    draft_probs = torch.zeros((1, gamma, vocab_size))
    for i, tok in enumerate([10, 20, 30, 40]):
        draft_probs[0, i, tok] = 1.0
        
    accepted_count, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.0,
    )
    
    assert accepted_count == 4
    assert next_tok.item() == 50


def test_rejection_sample_greedy_partial_accept():
    gamma = 4
    vocab_size = 100
    draft_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    
    # Target matches pos 0 and 1, but diverges at pos 2
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 10] = 1.0
    target_probs[0, 1, 20] = 1.0
    target_probs[0, 2, 99] = 1.0  # Divergence: target wants 99 instead of 30
    target_probs[0, 3, 40] = 1.0
    target_probs[0, 4, 50] = 1.0
    
    draft_probs = torch.zeros((1, gamma, vocab_size))
    for i, tok in enumerate([10, 20, 30, 40]):
        draft_probs[0, i, tok] = 1.0
        
    accepted_count, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.0,
    )
    
    assert accepted_count == 2
    assert next_tok.item() == 99


def test_server_args_spec_draft():
    args = [
        "--model", "Qwen/Qwen3.6-35B-A3B",
        "--spec-draft", "z-lab/Qwen3.6-35B-A3B-DFlash",
        "--spec-block-size", "7",
        "--spec-draft-dtype", "bfloat16",
    ]
    server_args, _ = parse_args(args)
    assert server_args.spec_draft_model == "z-lab/Qwen3.6-35B-A3B-DFlash"
    assert server_args.spec_block_size == 7
    assert server_args.spec_draft_dtype == "bfloat16"


def test_sampling_probs_temperature():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probs_greedy = _sampling_probs(logits, temperature=0.0)
    assert torch.allclose(probs_greedy, torch.tensor([[1.0, 0.0, 0.0]]))

    probs_temp = _sampling_probs(logits, temperature=1.0)
    assert probs_temp[0, 0] > probs_temp[0, 1] > probs_temp[0, 2]
    assert torch.allclose(probs_temp.sum(dim=-1), torch.tensor([1.0]))


def test_rejection_sample_stochastic():
    gamma = 2
    vocab_size = 10
    draft_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 1] = 1.0
    target_probs[0, 1, 2] = 1.0
    target_probs[0, 2, 3] = 1.0

    draft_probs = torch.zeros((1, gamma, vocab_size))
    draft_probs[0, 0, 1] = 1.0
    draft_probs[0, 1, 2] = 1.0

    accepted, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.7,
    )
    assert accepted == 2
    assert next_tok.item() == 3

