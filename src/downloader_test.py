import unittest
import downloader as dl
import time

NON_STR_ARGS = [None, -1, 2.8, TypeError]
VALID_STOCKS = ['AAPL', 'ekso', "tMus"]
INVALID_STOCKS = ['kitty']


class DownloaderTest(unittest.TestCase):

    def test_price_type(self):
        for case in NON_STR_ARGS:
            self.assertRaises(TypeError, dl.get_price, case)
            time.sleep(1)

    def test_price_value(self):
        BAD_ARGS = ["", "kitty"]
        for case in BAD_ARGS:
            self.assertRaises(ValueError, dl.get_price, case)
            time.sleep(1)

        for symb in INVALID_STOCKS:
            self.assertRaises(ValueError, dl.get_price, symb)

    def test_valid_stock(self):
        for symb in VALID_STOCKS:
            self.assertTrue(dl.is_valid_stock(symb))

    def test_invalid_stock(self):
        for symb in INVALID_STOCKS:
            self.assertFalse(dl.is_valid_stock(symb))
