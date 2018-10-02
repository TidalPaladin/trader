import mysql.connector as sql
import pandas as pd
from sqlalchemy import create_engine
import os
import glob

CWD = '/home/tidal/Dropbox/Software/trader/database/'
SYMBOLS_DIR = os.path.join(CWD, 'symbols')
SYMBOL_SQL_NAME = 'security'


# Read the CSV
def create_symbol_table(path):

    ALL_FILES = glob.glob(
        os.path.join(path, "*.csv")
    )

    result = pd.concat(
        (
            pd.read_csv(
                f,
                delimiter=',',
                usecols=[0, 1, 5, 6]
            )
            for f in ALL_FILES
        ),
        ignore_index=True
    )

    # Clean up indexing
    result.set_index('Symbol', inplace=True)
    result = result.drop_duplicates()
    result.sort_index(inplace=True)
    return result


def write_symbols_to_db(dataframe: pd.DataFrame, database):
    dataframe.to_sql(
        SYMBOL_SQL_NAME,
        con=database,
        if_exists='append'
    )
    pass


if __name__ == '__main__':
    engine = create_engine(
        'mysql+mysqlconnector://root:ChaseBowser1993!@127.0.0.1/trader')
    df = create_symbol_table(SYMBOLS_DIR)
    write_symbols_to_db(df, engine)
