"""Compatibility entry point; scoring lives in the installed package."""
import sys
from lr_cusum import event_scoring as implementation
globals().update({k:v for k,v in vars(implementation).items() if not k.startswith("__")})
if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
