from database import Database
from interface import IexDownloader, Downloader
from datetime import datetime, timedelta
import unittest

VALID_SYMBOLS = ['AAPL', 'SSC', 'EKSO', 'FB']
UNLISTED_SYMBOLS = ['ANUS', 'ROPE']
INVALID_SYMBOLS = ['', '100']
IEX_BATCH = "https://api.iextrading.com/1.0/stock/market/batch?symbols=%s&types=%s&range=%s"


class TestThis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = Database('root', 'ChaseBowser1993!')


class Symbols(TestThis):

    def test_add_symbols(self):
        result = self.db.refresh_symbol_table()
        self.assertGreater(len(result), 0)
        self.assertEquals('symbol', result.index.name.lower())

    def test_has_valids(self):
        for s in VALID_SYMBOLS:
            actual = self.db.has_symbol(s)
            self.assertTrue(actual)

    def test_has_invalids(self):
        for s in INVALID_SYMBOLS:
            actual = self.db.has_symbol(s)
            self.assertFalse(actual)

    def test_lookup_valids(self):
        for s in VALID_SYMBOLS:
            actual = self.db.lookup_symbol(s)
            self.assertEqual(actual.name, s)
            self.assertIn('name', actual.index.values)

    def test_lookup_invalids(self):
        for s in INVALID_SYMBOLS:
            actual = self.db.lookup_symbol(s)
            self.assertFalse(actual)


class UpdateHistory(TestThis):

    def test_result_layout(self):
        result = self.db.get_time_of_newest(VALID_SYMBOLS)
        self.assertIn('newest', result.columns)
        self.assertEqual('symbol', result.index.name)
        self.assertEqual(len(VALID_SYMBOLS), len(result))

        for case in VALID_SYMBOLS:
            self.assertIn(case, result.index)

    def test_defaults(self):
        exp_len = self.db.execute(
            'select count(*) from security').fetchall()[0][0]
        result = self.db.get_time_of_newest()
        self.assertEqual(exp_len, len(result))


class Merge(TestThis):

    @unittest.skip('none')
    def test_filter(self):

        df1 = self.db.dataframe('select * from history',
                                index_col=['symbol', 'start', 'end'])
        df2 = self.db.dataframe('select * from history', index_col='symbol')
        df1['junk'] = df1['low']
        df1['JUNK'] = df1['low']

        result1 = self.db.filter_dataframe('history', df1)
        result2 = self.db.filter_dataframe('history', df2)

        self.assertIn('junk', df1.columns)
        self.assertIn('JUNK', df1.columns)

        self.assertNotIn('junk', result1.columns)
        self.assertNotIn('JUNK', result1.columns)
        self.assertNotIn('junk', result2.columns)
        self.assertNotIn('JUNK', result2.columns)

        self.assertEqual(df1.index.names, result1.index.names)
        self.assertEqual(df2.index.name, result2.index.name)

    @unittest.skip('none')
    def test_merge(self):

        df1 = self.db.dataframe('select * from history',
                                index_col=['symbol', 'start', 'end'])
        df2 = self.db.dataframe('select * from history', index_col='symbol')
        df1['junk'] = df1['low']
        df1['JUNK'] = df1['low']

        self.db.merge('history', df1)

    def test_merge_dl(self):

        debug1 = '/home/tidal/Dropbox/Software/trader/data/tsla-6m.json'
        debug2 = '/home/tidal/Dropbox/Software/trader/data/aapl-6m.json'

        dl = IexDownloader('AAPL', timedelta(1), timedelta(25), url=debug2)
        dl.run()
        result = dl.result()
        self.db.merge('history', result)

        dl = IexDownloader('TSLA', timedelta(1), timedelta(25), url=debug1)
        dl.run()
        result = dl.result()
        self.db.merge('history', result)

        self.assertTrue(True)


class GetData(TestThis):

    def test_simple(self):
        result = self.db.get_history('AAPL')
        self.assertTrue(True)
