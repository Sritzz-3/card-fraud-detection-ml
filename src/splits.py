import pandas as pd

def time_split(df):
    d=df.sort_values('trans_date_trans_time',kind='stable'); n=len(d); a=int(.6*n); b=int(.8*n)
    return d.iloc[:a].copy(),d.iloc[a:b].copy(),d.iloc[b:].copy()
