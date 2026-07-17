"""Multiple-testing utilities for DML screening."""


def global_bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Return the unadjusted p-value cutoff for global Bonferroni control."""
    if n_tests <= 0:
        raise ValueError("n_tests must be positive")
    return alpha / n_tests
