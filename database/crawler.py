import pandas as pd
from database import Database
from interface import IexDownloader
from datetime import timedelta


if __name__ == '__main__':
    db = Database('root', 'ChaseBowser1993!')
    dl = IexDownloader('SSC', '5y', timedelta(365*8))
    dl.run()

    result = dl.result()
    db.merge('history', result)
