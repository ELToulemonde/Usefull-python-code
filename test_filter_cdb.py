from unittest import TestCase
import numpy as np
import pandas as pd
from .filter_cdb import fast_is_constant, fast_is_double, fast_is_bijection, filter_cdb


class TestFast_is_constant(TestCase):
    def test_fast_is_constant(self):
        self.assertTrue(fast_is_constant(np.array([])))  # Empty set is constant
        self.assertTrue(fast_is_constant(np.array(np.repeat(0, 1))))  # Small set
        self.assertFalse(fast_is_constant(np.append(np.array(np.repeat(0, 10)), np.array(np.nan))))  # Nan and constant
        self.assertTrue(fast_is_constant(np.array(np.repeat(np.float64(np.NaN), 10))))  # Only nans
        self.assertTrue(fast_is_constant(np.array(np.repeat(0, 9))))  # Only constant on sampling limits
        self.assertTrue(fast_is_constant(np.array(np.repeat(0, 11))))  # Only constant on sampling limits
        self.assertTrue(fast_is_constant(np.array(np.repeat("a", 11))))  # Work with string
        self.assertFalse(fast_is_constant(np.append(np.array(np.repeat(0, 10)), np.array(1))))   # Non constance at the end



class TestFast_is_double(TestCase):
    def test_fast_is_double(self):
        a = np.random.normal(size=100)
        b = a
        c = np.append(np.array([0, 0]), np.random.normal(size=98))
        d = a + 2
        e = np.repeat(np.nan, 1000)
        f = np.repeat(np.nan, 1000)
        g = np.random.choice(["a", "b"], 10)
        h = np.random.choice(["a", "b"], 10)
        self.assertTrue(fast_is_double(np.array([]), np.array([])))
        self.assertTrue(fast_is_double(a, b))
        self.assertFalse(fast_is_double(a, c))
        self.assertFalse(fast_is_double(a, d))
        self.assertTrue(fast_is_double(e, f))
        self.assertTrue(fast_is_double(g, g))
        self.assertFalse(fast_is_double(g, h))


class TestFast_is_bijection(TestCase):
    def test_fast_is_bijection(self):
        a = np.random.normal(size=100)
        b = a
        c = np.append(np.array([0, 0]), np.random.normal(size=98))
        d = a + 2
        e = np.array(["a", "b", "c"])
        f = np.array([0, 1, 0])
        g = np.array(["a", "b", "a"])
        h = np.array([np.nan, 1, np.nan])
        self.assertTrue(fast_is_bijection(a, b))
        self.assertFalse(fast_is_bijection(a, c))
        self.assertTrue(fast_is_bijection(a, d))
        self.assertFalse(fast_is_bijection(e, f))
        self.assertTrue(fast_is_bijection(f, g))  #
        self.assertTrue(fast_is_bijection(f, h))  # Bijection with nan is still a bijection
        self.assertTrue(fast_is_bijection(np.array([]), np.array([])))  # Empty is a bijection of empty
        self.assertFalse(fast_is_bijection(np.array([1]), np.array([])))  # Dimension mismatch


class TestFilter_cdb(TestCase):
    def test_filter_cdb(self):
        a = np.random.normal(size=100)
        b = a
        c = np.append(np.array([0, 0]), np.random.normal(size=98))
        d = a + 2
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "constant": np.repeat(0, len(d))})
        self.assertEqual(filter_cdb(df.copy(), "c").shape[1], 4)
        self.assertEqual(filter_cdb(df.copy(), "cd").shape[1], 3)
        self.assertEqual(filter_cdb(df.copy(), "cdb").shape[1], 2)
