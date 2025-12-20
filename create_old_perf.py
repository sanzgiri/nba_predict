import pandas as pd
import numpy as np
import glob
import re
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import json

abbr = {"ATLANTA HAWKS": "ATL",
         "BOSTON CELTICS": "BOS",
         "BROOKLYN NETS": "BRK",
         "CHARLOTTE HORNETS": "CHO",
         "CHICAGO BULLS": "CHI",
         "CLEVELAND CAVALIERS": "CLE",
         "DALLAS MAVERICKS": "DAL",
         "DENVER NUGGETS": "DEN",
         "DETROIT PISTONS": "DET",
         "GOLDEN STATE WARRIORS": "GSW",
         "HOUSTON ROCKETS": "HOU",
         "INDIANA PACERS": "IND",
         "LOS ANGELES CLIPPERS": "LAC",
         "LOS ANGELES LAKERS": "LAL",
         "MEMPHIS GRIZZLIES": "MEM",
         "MIAMI HEAT": "MIA",
         "MILWAUKEE BUCKS": "MIL",
         "MINNESOTA TIMBERWOLVES": "MIN",
         "NEW ORLEANS PELICANS": "NOP",
         "NEW YORK KNICKS": "NYK",
         "OKLAHOMA CITY THUNDER": "OKC",
         "ORLANDO MAGIC": "ORL",
         "PHILADELPHIA 76ERS": "PHI",
         "PHOENIX SUNS": "PHO",
         "PORTLAND TRAIL BLAZERS": "POR",
         "SACRAMENTO KINGS": "SAC",
         "SAN ANTONIO SPURS": "SAS",
         "TORONTO RAPTORS": "TOR",
         "UTAH JAZZ": "UTA",
         "WASHINGTON WIZARDS": "WAS"}
         
flist = glob.glob("standalone/kelly_2021*.csv")
df = pd.DataFrame()
prev_days = sorted(flist)[:-1]

for f in prev_days:
    dfi = pd.read_csv(f)
    
    date = re.match('standalone/kelly_(.*).csv', f).groups()[0]
    dfi.insert(0, 'date', date)
    
    y, m, d = date.split('-')
    box_scores = client.team_box_scores(day=int(d), month=int(m), year=int(y), output_type=OutputType.JSON)
    my_json = json.loads(box_scores)
    
    nitems = len(my_json)
    wdict = {}
    for i in range(nitems):
#        print(my_json[i]['team'], my_json[i]['outcome'])
        team_code = abbr[my_json[i]['team']]
        is_winner = int(my_json[i]['outcome'] == "WIN")
        wdict[team_code] = is_winner

    dfi['pred_winner'] = 0
    dfi['is_winner'] = -1
    
    for i in range(len(dfi)):
        if dfi['win_prob'].iloc[i] >= 0.5:
            dfi.loc[i, 'pred_winner'] = 1 
            dfi.loc[i, 'pred_team'] = dfi['team'].iloc[i]
        else:
            dfi.loc[i, 'pred_team'] = dfi['opponent'].iloc[i]
        try:
            dfi.loc[i, 'is_winner'] = wdict[dfi['team'].iloc[i]]
        except KeyError:
            print(f"Date {date}: Game Canceled for {dfi['team'].iloc[i]}")
            
    dfi['win_prob'] = dfi['win_prob'].apply(lambda x: round(x,3))
    dfi['odds'] = dfi['odds'].apply(lambda x: round(x,3))
    dfi['kelly_weights'] = dfi['kelly_weights'].apply(lambda x: round(x,3))
    
    dfi = dfi[['date','team','opponent','location','win_prob','pred_winner', 'pred_team', 'is_winner','odds','kelly_weights']]
    df = df.append(dfi)
    
df.to_csv('perf/historic_perf.csv', index=None)