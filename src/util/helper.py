import pandas as pd


@staticmethod
def printpd(o):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(o)

@staticmethod
def print_filepath_n_rows(filepath, n):
    with open(filepath) as f:
        head = [next(f) for x in range(n)]
        print(head)