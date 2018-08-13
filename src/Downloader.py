import requests
from enum import Enum
import json

IEX_URL_REGEX = "https://api.iextrading.com/1.0/stock/%s/%s/%s"


def get_history(stock_symbol, timeframe, output_path=""):
    """
    Fetch price histories using the IEX trading API
    Parameters
    ----------
    stock_symbol : string
        The symbol for the stock to fetch, ie 'AAPL'

    timeframe : string
        The IEX allowed timeframe to fetch, ie '6m'.

    output_path : string
        If supplied, a file path where the JSON data will be written

    Return
    ---------
    json : The historical price data
    """

    if not timeframe:
        raise ValueError('Timeframe was not specified')
    elif not stock_symbol:
        raise ValueError('Stock symbol was not specified')

    r = iex_raw_query(stock_symbol, 'chart', timeframe if timeframe else None)

    # If output path is given, write JSON to file before return
    if output_path:
        write_json(r.json(), output_path)

    return r.json()


def get_price(stock_symbol):
    """
    Query IEX for the current market price of the given symbol. Price may be delayed.
    See get_quote() for more comprehensive real time options

    Paramters
    ---
    stock_symbol : string
        Symbol of the stock to fetch
    """
    return iex_raw_query(stock_symbol, 'price').json()


def get_quote(stock_symbol):
    """
    Get quote data for a given stock

    Parameters
    ---
    stock_symbol : string
        Stock exchange symbol for the stock to query
    """
    return iex_raw_query(stock_symbol, 'quote').json()


def write_json(json_data, output_path):
    """
    Write the supplied JSON to an output file

    Parameters
    ---

    json_data : JSON
        The data to be written

    output_path : string
        The path to the destination file
    """

    if not json_data:
        raise ValueError('Empty JSON parameter')
    elif not output_path:
        raise ValueError('Empty output path')

    with open(output_path, 'w') as outfile:
        json.dump(json_data, outfile)


def read_json(file_path):
    """
    Reads the supplied JSON file and returns the corresponding JSON object

    Paramters
    ---
    file_path : string
        Path to the JSON file to read
    """

    with open(file_path, 'r') as in_file:
        result = json.load(in_file)
    return result


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

    url = IEX_URL_REGEX % (stock_symbol, query_type, timeframe)
    result = requests.get(url, params=param_dict)

    if not result.status_code == 200:
        raise IOError('Got abnormal error code %i' % result.status_code)
    return result
