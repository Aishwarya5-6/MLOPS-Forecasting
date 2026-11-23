# torch and xgboost each ship their own libomp. On macOS, if xgboost's copy loads first,
# training a torch model in the same process segfaults. Importing torch first avoids it.
# Only the test suite loads both; train.py / sweep / pyfunc load one model type per process.
import torch  # noqa: F401
