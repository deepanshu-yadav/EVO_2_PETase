#!/usr/bin/env python3
"""Module 1 Test Runner and Diagnostic Audit.

Executes test_module_1_sampling.py, collects diagnostics on:
  - Cluster balancing verification
  - Curriculum Tm monotonicity
  - Translation fidelity
  - Spacer invariants
and prints a clean audit report.
"""
import io
import pathlib
import sys
import time
import unittest

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import test_module_1_sampling


def main():
    print("=" * 78)
    print("RUNNING MODULE 1 SAMPLING VERIFICATION TESTS")
    print("=" * 78)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_module_1_sampling.TestModule1Sampling)

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("-" * 78)
    print(f"Tests run: {result.testsRun} | Failures: {len(result.failures)} | Errors: {len(result.errors)} in {elapsed:.2f}s")
    if result.wasSuccessful():
        print(">>> ALL MODULE 1 SAMPLING TESTS PASSED SUCCESSFULLY! <<<")
        print("=" * 78)
        sys.exit(0)
    else:
        print(">>> SOME TESTS FAILED! Check traceback above. <<<")
        print("=" * 78)
        sys.exit(1)


if __name__ == "__main__":
    main()
