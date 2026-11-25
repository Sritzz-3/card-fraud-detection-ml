"""Seeded synthetic transaction generator; intentionally no external data access."""
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ['grocery_pos','gas_transport','shopping_pos','shopping_net','entertainment','food_dining','health_fitness','home','travel','misc_pos','misc_net']
MERCHANTS = [f'merchant_{i:04d}' for i in range(600)]
JOBS = ['engineer','teacher','nurse','manager','driver','analyst','sales','technician','artist','clerk']

def generate_transactions(n=1_000_000, seed=42, n_cards=1500):
    rng=np.random.default_rng(seed)
    cards=np.arange(1000000000000000,1000000000000000+n_cards)
    card=rng.choice(cards,n)
    start=np.datetime64('2022-01-01T00:00:00'); seconds=rng.integers(0,730*86400,n)
    times=start+seconds.astype('timedelta64[s]')
    card_ix=card-cards[0]
    home_lat= rng.uniform(25,48,n_cards); home_lon=rng.uniform(-124,-70,n_cards)
    lat=home_lat[card_ix]+rng.normal(0,.35,n); lon=home_lon[card_ix]+rng.normal(0,.45,n)
    cat=rng.choice(CATEGORIES,n); amount=rng.lognormal(3.1,.9,n)
    fraud=rng.random(n)<0.005
    # Imperfect patterns overlap legitimate behavior by design.
    high=rng.random(n)<.62; amount=np.where(fraud & high, amount*rng.uniform(3,12,n),amount)
    cat=np.where(fraud & (rng.random(n)<.45),rng.choice(['shopping_net','misc_net','misc_pos'],n),cat)
    lat=np.where(fraud & (rng.random(n)<.35),rng.uniform(25,48,n),lat)
    lon=np.where(fraud & (rng.random(n)<.35),rng.uniform(-124,-70,n),lon)
    order=np.argsort(times,kind='stable')
    df=pd.DataFrame({'trans_date_trans_time':times,'cc_num':card,'merchant':rng.choice(MERCHANTS,n),'category':cat,'amt':amount.round(2),'gender':rng.choice(['F','M'],n),'city_pop':rng.integers(500,1000000,n),'job':rng.choice(JOBS,n),'dob':(np.datetime64('1960-01-01')+rng.integers(0,18000,n).astype('timedelta64[D]')),'lat':lat,'long':lon,'merch_lat':lat+rng.normal(0,.2,n),'merch_long':lon+rng.normal(0,.2,n),'is_fraud':fraud.astype('int8')})
    return df.iloc[order].reset_index(drop=True)

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--rows',type=int,default=1_000_000); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',default='data/raw/transactions.csv'); a=p.parse_args()
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); generate_transactions(a.rows,a.seed).to_csv(path,index=False)
if __name__=='__main__': main()
