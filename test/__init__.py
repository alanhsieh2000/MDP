# test/__init__.py

import unittest
from .test_agent import *

def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    tests = (TestAgent, TestMonteCarlo)
    for test_class in tests:
        test_cases = loader.loadTestsFromTestCase(test_class)
        suite.addTests(test_cases)
    return suite