__author__ = "TOULEEM"
__project__ = "DADGP"


from unittest import TestCase
from .argument_parser import encode_arguments


class TestEncode_arguments(TestCase):
    def test_empty_arg(self):
        self.assertEqual(encode_arguments(), [])

    def test_simple_arg(self):
        self.assertEqual(encode_arguments(a="a"), ["a=a=character"])

    def test_complex_arg(self):
        self.assertEqual(encode_arguments(sep="|"), ["sep=%7C=character"])

    def test_typing(self):
        self.assertEqual(encode_arguments(a=int(1)), ["a=1=integer"])
        self.assertEqual(encode_arguments(a=float(1)), ["a=1.0=numeric"])
        self.assertEqual(encode_arguments(a=True), ["a=True=logical"])

    def test_multiple_args(self):
        self.assertEqual(encode_arguments(a="a", sep="|"), ["a=a=character", "sep=%7C=character"])

