from database import Database
from interface import Downloader, IexDownloader
import unittest
from datetime import timedelta, datetime

VALID_SYMBOLS = ['AAPL', 'SSC', 'EKSO', 'FB']
UNLISTED_SYMBOLS = ['ANUS', 'ROPE']
INVALID_SYMBOLS = ['', '100']
IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"


class TestThis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database('root', 'ChaseBowser1993!')


class IexTest(TestThis):

    def test_local(self):
        dl = IexDownloader('aapl', timedelta(1), timedelta(25))
        dl.run()
        result = dl.result()

        expected_index = ['symbol', 'start', 'end']
        for val in expected_index:
            self.assertIn(val, result.index.names)

        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
