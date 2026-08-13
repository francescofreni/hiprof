"""Hi Prof 👋: High-Probability Falsifier.

The main public interface is:

- HPFalsifier: check whether a candidate observational
    formula identifies the target interventional distribution,
    by evaluating both on randomly sampled linear Gaussian
    models.


Example:
    >>> from hiprof import HPFalsifier
    >>> falsifier = HPFalsifier(
    ...     graph="T -> M; M -> Y; T <-> Y",
    ...     treatments="T",
    ...     outcomes="Y",
    ... )
    >>> formula = '''
    ... sum_{M} {
    ...     p(M | T)
    ...     sum_{T'} { p(Y | M, T') p(T') }
    ... }
    ... '''
    >>> falsifier.check(formula)
    True
    False-acceptance bound: 5.421e-18
"""

from .identification import IDAlgorithm
from .verification.falsifier import CheckResult, HPFalsifier


__all__ = [
    "CheckResult",
    "HPFalsifier",
    "IDAlgorithm",
]
