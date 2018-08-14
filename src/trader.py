import Downloader as dl
dl.download_symbols('symbols.txt')
symb = dl.get_symbol_list('symbols.txt')
print(symb)
