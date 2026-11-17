"""Lightweight PyTorch stub for CPU-only API contract tests when torch is absent.

The actual model is never loaded: _run_inference is monkeypatched in tests.
"""
import importlib.util
import sys
import types

if importlib.util.find_spec("torch") is None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch
