"""``headroom.summarise``: the figures the temperature test reports."""

from headroom import summarise


def answer(edits: int, reward: float, looped: int = 0, capped: int = 0) -> dict:
    return {"edits": edits, "chars": 100, "reward": reward, "looped": looped, "capped": capped, "tokens": 10}


def test_best_sample_by_reward_and_by_cer_against_greedy() -> None:
    greedy = {"a": answer(10, 0.9), "b": answer(20, 0.8)}
    rows = [
        {"id": "a", "answers": [answer(6, 0.5), answer(12, 0.95)]},  # the reward prefers the worse reading
        {"id": "b", "answers": [answer(30, 0.6), answer(40, 0.1, looped=1)]},
    ]
    found = summarise(rows, greedy)
    assert found["greedy_cer"] == 15.0
    assert found["mean_sample_cer"] == 22.0
    assert found["best_by_reward_cer"] == 21.0
    assert found["best_by_cer"] == 18.0
    assert found["pages_a_sample_beats_greedy"] == 1
    assert found["pages_best_by_reward_beats_greedy"] == 0
    assert found["answers_looped"] == 0.25
    assert (found["clean_pages"], found["clean_greedy_cer"], found["clean_best_by_cer"]) == (1, 10.0, 6.0)
