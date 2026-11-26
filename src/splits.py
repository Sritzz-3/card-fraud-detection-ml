import pandas as pd

def split_train_validation(train, test=None, validation_fraction=.20):
    """Use last chronological slice of fraudTrain as validation; preserve fraudTest as test."""
    train = train.sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)
    cut = int(len(train)*(1-validation_fraction))
    fit, validation = train.iloc[:cut].copy(), train.iloc[cut:].copy()
    if test is None:
        test = validation.iloc[0:0].copy()
    else:
        test = test.sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)
    if len(fit) and len(validation):
        assert fit.trans_date_trans_time.max() < validation.trans_date_trans_time.min(), 'Train/validation dates overlap'
    if len(validation) and len(test):
        assert validation.trans_date_trans_time.max() < test.trans_date_trans_time.min(), 'Validation/test dates overlap'
    return fit, validation, test

def time_split(df):
    """Compatibility 60/20/20 chronological split used by utility tests."""
    d=df.sort_values('trans_date_trans_time',kind='stable'); n=len(d)
    return d.iloc[:int(.6*n)].copy(),d.iloc[int(.6*n):int(.8*n)].copy(),d.iloc[int(.8*n):].copy()
