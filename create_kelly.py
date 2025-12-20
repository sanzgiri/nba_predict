import pandas as pd
import glob
import re

flist = glob.glob("standalone/kelly_2021*.csv")
df = pd.DataFrame()

for f in sorted(flist):
    dfi = pd.read_csv(f)
    date = re.match('standalone/kelly_(.*).csv', f).groups()[0]
    #dfi['date'] = date
    dfi.insert(0, 'date', date)
    dfi['win_prob'] = dfi['win_prob'].apply(lambda x: round(x,3))
    dfi['exp_profit'] = dfi['exp_profit'].apply(lambda x: round(x,3))
    dfi['odds'] = dfi['odds'].apply(lambda x: round(x,3))
    dfi['kelly_weights'] = dfi['kelly_weights'].apply(lambda x: round(x,3))
    df = df.append(dfi)
    
df.to_csv('perf/historic_kelly.csv', index=None)