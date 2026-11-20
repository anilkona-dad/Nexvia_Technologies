"""Unified Test Runner for Causal Inference & Experimentation Platform."""

import sys
from backend.tests.test_causal_platform import (
    test_power_calculator,
    test_frequentist_tests,
    test_sequential_tests,
    test_bayesian_ab,
    test_causal_observational_psm,
    test_causal_observational_did,
    test_causal_observational_scm,
    test_plain_english_explainer
)

TESTS = [
    ("Milestones 1 & 2: Power & Sample Size Calculator", test_power_calculator),
    ("Milestone 3: Frequentist Tests & Multiple Correction", test_frequentist_tests),
    ("Milestone 4: Safe Sequential Testing (mSPRT & Alpha Spending)", test_sequential_tests),
    ("Milestone 5: Bayesian A/B Engine (Beta-Binomial & Loss)", test_bayesian_ab),
    ("Milestone 6: Observational PSM (LaLonde & DoWhy)", test_causal_observational_psm),
    ("Milestone 6: Observational DiD (Card & Krueger 1994)", test_causal_observational_did),
    ("Milestone 6: Observational SCM (California Prop 99)", test_causal_observational_scm),
    ("Milestone 7: Plain-English Decision Engine", test_plain_english_explainer)
]

def run():
    passed = 0
    failed = 0
    print("\n" + "="*70)
    print("  CAUSAL & EXPERIMENTATION PLATFORM - VERIFICATION TEST SUITE")
    print("="*70)

    for name, fn in TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("="*70)
    print(f"  Results: {passed} Passed, {failed} Failed")
    print("="*70 + "\n")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run()
