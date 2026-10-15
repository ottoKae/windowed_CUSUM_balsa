"""Compatibility import; implementation is installed in lr_cusum."""
import sys
from lr_cusum import b20_probability_estimation as implementation
globals().update({k: v for k, v in vars(implementation).items() if not k.startswith('__')})
sys.modules[__name__] = implementation
