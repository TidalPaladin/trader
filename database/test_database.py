import database as db
import unittest
from datetime import timedelta as td
import numpy as np

VALID_SYMBOLS = ['AAPL', 'SSC', 'EKSO', 'FB']
UNLISTED_SYMBOLS = ['ANUS', 'ROPE']
INVALID_SYMBOLS = ['', '100']
IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"


class RequestBuilder(unittest.TestCase):

    def test_single_symbol(self):
        test_type = 'chart'
        for case in VALID_SYMBOLS:
            url = db.build_batch_request([case], [test_type], td(3))
            expected = IEX_BATCH % (case, test_type, '1m')
            self.assertEquals(expected, url)

    def test_group_symbol(self):
        test_type = 'chart'
        url = db.build_batch_request(VALID_SYMBOLS, [test_type], td(3))
        expected = IEX_BATCH % (','.join(VALID_SYMBOLS), test_type, '1m')
        self.assertEquals(expected, url)

    def test_last(self):
        test_type = 'chart'
        last = 5
        url = db.build_batch_request(
            VALID_SYMBOLS, [test_type], td(3), last=last)
        new_batch = IEX_BATCH + ("&last=%i" % last)
        expected = new_batch % (','.join(VALID_SYMBOLS), test_type, '1m')
        self.assertEquals(expected, url)


class TimedeltaConverter(unittest.TestCase):

    def test_simple(self):
        n = 10
        cases = {
            '1d': [td(0, int(x)) for x in np.random.randint(1, 3600*8, n)],
            '1m': [td(int(x), int(x)) for x in np.random.randint(1, 29, n)],
            '3m': [td(int(x), int(x)) for x in np.random.randint(29, 91, n)],
            '6m': [td(int(x), int(x)) for x in np.random.randint(91, 181, n)],
            '1y': [td(int(x), int(x)) for x in np.random.randint(181, 366, n)],
            '2y': [td(int(x), int(x)) for x in np.random.randint(366, 366*2+1, n)],
            '5y': [td(int(x), int(x)) for x in np.random.randint(366*2+1, 1024, n)],
        }

        for k, vals in cases.items():
            for v in vals:
                self.assertEquals(k, db._timedelta_to_iex_period(v), msg=k)

    def test_edge(self):
        cases = {
            '1d': [td(0, 0), td(0, -260)],
            '5y': [td(365*10)]
        }

        for k, vals in cases.items():
            for v in vals:
                self.assertEquals(k, db._timedelta_to_iex_period(v), msg=k)
