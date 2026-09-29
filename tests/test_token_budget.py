from utils.token_budget import TokenBudget


def test_token_budget_select_never_exceeds_limit():
    budget = TokenBudget(model="gpt-4.1-mini")
    selected = budget.select(
        [
            ("a", "alpha " * 20),
            ("b", "beta " * 20),
        ],
        max_tokens=12,
    )
    assert selected
    assert sum(budget.count(text) for _, text in selected) <= 12


def test_token_budget_batches_each_fit_limit():
    budget = TokenBudget(model="gpt-4.1-mini")
    batches = budget.batches(
        [
            ("a", "alpha " * 10),
            ("b", "beta " * 10),
            ("c", "gamma " * 10),
        ],
        max_tokens=12,
    )
    assert len(batches) >= 2
    assert all(
        sum(budget.count(text) for _, text in batch) <= 12
        for batch in batches
    )
