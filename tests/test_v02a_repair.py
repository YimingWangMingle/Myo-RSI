import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.rsi_v02a.adapt import check_budget
from experiments.rsi_v02a.registry import cohorts,source
from experiments.rsi_v02a.diagnosis import selected

class RepairContract(unittest.TestCase):
    def test_invalid_cost(self):
        with self.assertRaises(ValueError):check_budget("B3",240000,1200000)
        check_budget("B1",0,1200000)
        check_budget("B3",240000,960000)

    def test_no_leak(self):
        with patch("experiments.rsi_v02a.registry.read",return_value={"cohorts":{"a":[0,10],"b":[10,20]}}):
            with self.assertRaises(ValueError):cohorts()

    def test_invalid_checkpoint(self):
        with patch("experiments.rsi_v02a.registry.sha",return_value="wrong"):
            with self.assertRaises(ValueError):source(11)

    def test_case_selection(self):
        with self.assertRaises(ValueError):selected([dict(success=0,goal=1,error=.1)])
