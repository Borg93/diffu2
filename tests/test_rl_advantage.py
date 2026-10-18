"""Unit tests for the RL group-relative advantage (the one piece of src/diffu/rl that is pure logic)."""

import numpy as np
import pytest

from diffu.rl import PerPromptAdvantage


def test_sft_best_of_group() -> None:
    """SFT advantage = 1.0 for the max-reward sample in each prompt's group, else 0.0."""
    out = PerPromptAdvantage(mode="sft")(["a", "a", "b"], [1.0, 3.0, 2.0])
    assert out.tolist() == [0.0, 1.0, 1.0]  # a: 3 wins; b: lone sample wins


def test_grpo_normalized() -> None:
    """GRPO advantage = (reward − group_mean) / std → standardized within the group."""
    out = PerPromptAdvantage(mode="grpo", global_std=False)(["a", "a"], [0.0, 2.0])
    assert out[0] < 0 < out[1]
    np.testing.assert_allclose(out, [-1.0, 1.0], atol=1e-3)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        PerPromptAdvantage()(["a"], [1.0, 2.0])


def test_bad_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        PerPromptAdvantage(mode="ppo")
