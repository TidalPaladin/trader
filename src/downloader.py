import requests
import ftplib as ftp
import os

import json
import csv

IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"
NASDAQ_FTP_HOST = "ftp.nasdaqtrader.com"

SYMBOL_FILE = 'symbols.txt'
SYMBOL_LIST = []


def get_history(stock_symbol, timeframe, output_path=""):
    """Fetch IEX market data on 'stock_symbol' over the given timeframe
    If 'output_path' is specified, the JSON will be written to the file
    """

    # Preconditions
    precond_check_str_type("stock_symbol", stock_symbol)
    precond_check_str_type("timeframe", timeframe)
    if output_path and type(output_path) is not str:
        raise TypeError('output path given but not of type str')

    json_result = iex_raw_query(stock_symbol, 'chart', timeframe)

    # If output path is given, write JSON to file before return
    if output_path:
        with open(output_path, 'w') as fp:
            json.dump(json_result, fp)
            fp.close()

    return json_result


def get_price(stock_symbol):
    """
    Query IEX for the current market price of stock 'stock_symbol'. Price may be delayed.
    See get_quote() for more comprehensive real time options. Returns a
    """
    if not is_valid_stock(stock_symbol):
        raise ValueError('stock_symbol %s does not exist' % stock_symbol)

    return iex_raw_query(stock_symbol, 'price').json()


def get_quote(stock_symbol):
    """Get quote data for stock 'stock_symbol'"""
    if not is_valid_stock(stock_symbol):
        raise ValueError('stock_symbol %s does not exist' % stock_symbol)

    return iex_raw_query(stock_symbol, 'quote').json()


def iex_raw_query(stock_symbol, query_type, timeframe="", param_dict=[]):
    """
    Make a raw query to the IEX API, returning a requests object with JSON data

    Paramters
    ---
    stock_symbol : string
        Stock exchange symbol for the stock to query, ie 'AAPL'
    query_type : string
        Type of query to be made as allowed by IEX API, ie 'chart' or 'quote'
    timeframe : string
        If applicable, the timeframe to query, ie '6m'
    param_dict : dictionary
        If desired, a list of key value pairs to be included as parameters with the API call
    """

    if not is_valid_stock(stock_symbol):
        raise ValueError('Unknown stock symbol %s' % stock_symbol)

    url = IEX_URL_REGEX % (stock_symbol, query_type, timeframe)
    result = requests.get(url, params=param_dict)

    if not result.status_code == 200:
        raise IOError('Got abnormal error code %i' % result.status_code)
    return result


def download_symbols(output_path):
    """
    Fetch a list of all listed stock symbols and dump the list into an output file.
    Output file is unmodified from NASDAQ, but should be parsed with nasdaq_csv_to_newline()

    Parameters
    ---
    output_path : string
        File path where the resultant symbols will be stored

    Post
    ---
    'output_path' file created or overwritten with updated NASDAQ symbols
    """

    SYMBOL_DIR = "SymbolDirectory"
    SYMBOL_FILE = "nasdaqlisted.txt"

    # Download CSV symbol file over FTP
    session = ftp.FTP(NASDAQ_FTP_HOST)
    session.login()
    session.cwd(SYMBOL_DIR)
    with open(output_path, 'wb') as csvfile:
        session.retrbinary("RETR " + SYMBOL_FILE, csvfile.write)
        csvfile.close()
    session.quit()

    # Convert to newline separated list, write to file
    DELIMITER = '\n'
    symbols = nasdaq_csv_to_newline(output_path)
    with open(output_path, 'w') as fstream:
        for symbol in symbols:
            fstream.write(symbol + DELIMITER)

    return symbols


def nasdaq_csv_to_newline(source_file):
    """
    Parse a CSV symbol file downloaded from the NASDAQ FTP server into a list
    of symbols

    Parameters
    ---
    source_file : string
        Path to the CSV file downloaded from NASDAQ
    """

    precond_check_str_type("source_file", source_file)
    if not os.path.exists(source_file):
        raise FileNotFoundError('Missing file %s' % source_file)

    result = []
    with open(source_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        CSV_SYMBOL_INDEX = 0
        for row in reader:
            result.append(row[CSV_SYMBOL_INDEX])

    # Remove the 'Symbol' and file creation time rows
    del result[-1]
    del result[0]

    return result


def iex_crawl(symbol_list, seconds_between):

    if type(symbol_list) is not list:
        raise TypeError('symbol_list must be a list')
    if not symbol_list:
        raise ValueError('symbol_list must not be empty')
    for symbol in symbol_list:
        if not is_valid_stock(symbol):
            raise ValueError('symbol_list contained invalid stock %s' % symbol)

    # Ensure symbol directory exists
    ROOT_DIR = './symbols'
    if not os.path.isdir(ROOT_DIR):
        os.makedirs('ROOT_DIR')


def precond_check_str_type(str_name, str_value):
    """Check if variable with name str_name with value str_value is a string"""
    if type(str_value) is not str:
        raise TypeError(str_name + ' is not of type str')
    if not len(str_value):
        raise ValueError(str_name + ' must not be empty')


def is_valid_stock(symbol):
    """Check if a stock symbol is NASDAQ listed"""
    global SYMBOL_LIST
    if not SYMBOL_LIST:
        SYMBOL_LIST = open(SYMBOL_FILE).read().splitlines()
    precond_check_str_type('symbol', symbol)
    return symbol.upper() in SYMBOL_LIST
