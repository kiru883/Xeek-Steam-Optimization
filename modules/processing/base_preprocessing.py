import pandas as pd


def remove_bad_variables(df):
    """Checks dataframe for variables and drops
    them if present.

    df: dataframe
    """
    col_names = list(df.columns)
    for i in ['RMNG_OIL_H', 'GAS_H']:
        if i in col_names:
            df = df.drop(i, axis=1)
    return df