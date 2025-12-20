
import numpy as np
import pandas as pd
#import datetime as dt
from datetime import date, datetime, timezone, timedelta

import os
import ast
import pandas as pd
import re

def expected_revenue(wagers,odds,predicts):
    wagers = pd.Series(wagers)
    rev = (wagers * odds * predicts)
    return((wagers-rev).sum())

def revenue_long(wagers,odds,predicts):
    wagers = pd.Series(wagers)
    rev = (wagers * odds * predicts)
    return(rev-wagers)
    
def odds_conversion(line):
    if line < 0:
        odds = (-line + 100) / -line
    else:
        odds = (line + 100) / 100
    return(odds)
    
def kelly(odds,predicts):
    b = odds - 1
    q = 1 - predicts
    f = (b*predicts) - q
    f = f / b
    return(f)

roto_dict = {'PHO': 'PHX',
            'BRO' : 'BRK',
            'SAN' : 'SAS',
            'NY' : 'NYK',
            'CHR' : 'CHO',
            'GS' : 'GSW',
            'NOR': 'NOP'}

abbr2 = {"ATLANTA HAWKS": "ATL",
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
         "PHOENIX SUNS": "PHX",
         "PORTLAND TRAIL BLAZERS": "POR",
         "SACRAMENTO KINGS": "SAC",
         "SAN ANTONIO SPURS": "SAS",
         "TORONTO RAPTORS": "TOR",
         "UTAH JAZZ": "UTA",
         "WASHINGTON WIZARDS": "WAS"}


# helper functions for evaluating a portfolio or using a strategy
def eval_portfolio(w, df):
    outcome = df.win_prob.apply(simulate_game)
    revenue = revenue_long(w, df['odds'], outcome)
    return revenue


def softmax_strat_pos_ev_only(ev, temp):
    ev = pd.Series(ev).copy()
    ev.loc[ev < 0] = np.nan
    w = np.exp(ev / temp)
    w = w / np.sum(w)
    w.loc[w.isna()] = 0
    return w


def softmax_strat_all(ev, temp):
    ev = pd.Series(ev).copy()
    w = np.exp(ev / temp)
    w = w / np.sum(w)
    w.loc[w.isna()] = 0
    return w


def kelly_strat(ev):
    ev = pd.Series(ev).copy()
    ev.loc[ev < 0] = np.nan
    w = ev.copy()
    w.loc[w.isna()] = 0
    return w

def eval_real_portfolio(w, df):
    outcome = df.winner.values
    revenue = revenue_long(w, df['odds'], outcome)
    return revenue

def get_todays_weights():

    today = date.today().strftime("%Y-%m-%d")
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y_%m_%d")

    cmd = f"wget -O odds.txt \"https://www.rotowire.com/basketball/tables/odds.php?date={today}&odds=stats\""
    # print(cmd)
    os.system(cmd)
    with open("odds.txt", "r") as myfile:
        data = myfile.read().replace('\n', '')
    data2 = data[1:-1]
    mydict = ast.literal_eval(data2)
    df = pd.DataFrame.from_dict(mydict)
    df['favourite'] = df['favourite'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df['underdog'] = df['underdog'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df = df.rename(columns={"favourite": "favorite",
                            "fml": "favorite_moneyline",
                            "uml": "underdog_moneyline",
                            "ou": "over-under"})

    df = df[['time', 'favorite', 'underdog', 'favorite_moneyline', 'underdog_moneyline', 'line', 'over-under']]

    lines = pd.DataFrame()
    for i in range(len(df)):
        lines.loc[i, 'date'] = pd.to_datetime(today)
        lines.loc[i, 'team'] = df.favorite.iloc[i]
        lines.loc[i, 'line'] = df.favorite_moneyline.iloc[i]
        lines.loc[i+len(df), 'date'] = pd.to_datetime(today)
        lines.loc[i+len(df), 'team'] = df.underdog.iloc[i]
        lines.loc[i+len(df), 'line'] = df.underdog_moneyline.iloc[i]

    lines['team'] = lines['team'].apply(lambda x: roto_dict[x] if x in roto_dict.keys() else x)

    lines_empty = lines[lines.line == ''].copy()
    lines_full = lines[lines.line != ''].copy()
    lines_empty['line'] = 0
    lines_full['line'] = lines_full['line'].astype(int)
    lines = pd.concat([lines_full, lines_empty], axis=0)

    games = pd.read_csv(f'game_predictions_{tomorrow}.csv')
    games['team1'] = games['HOME_team'].apply(lambda x: abbr2[x])
    games['team2'] = games['AWAY_team'].apply(lambda x: abbr2[x])
    games['date'] = pd.to_datetime(today)

    games = games[['date', 'team1', 'team2', 'HOME_wins_p', ]].copy()
    games['team2_win_prob'] = 1.0 - games['HOME_wins_p']
    g1 = games[['date', 'team1', 'HOME_wins_p']].copy()
    g1.columns = ['date', 'team', 'win_prob']
    g2 = games[['date', 'team2', 'team2_win_prob']].copy()
    g2.columns = ['date', 'team', 'win_prob']
    df_g = pd.concat([g1, g2])

    df = df_g.merge(lines, on=['date', 'team'], how='left')
    df['odds'] = df['line'].apply(odds_conversion)
    df['exp_profit'] = revenue_long(np.repeat(1, df.shape[0]), df['odds'], df['win_prob'])
    df['kelly'] = kelly(df['odds'], df['win_prob'])

    # apply the strategies
    kellys = pd.DataFrame(kelly_strat(df['kelly']))
    softmaxx = pd.DataFrame(softmax_strat_pos_ev_only(df['exp_profit'], 0.5))

    kellys.columns = ['kelly_weights']
    softmaxx.columns = ['softmax_weights']

    final = pd.concat([kellys, softmaxx, df], axis=1)

    budget = 10
    final['kelly_weights'] = final['kelly_weights'] * budget
    final['softmax_weights'] = final['softmax_weights'] * budget

    final = final.sort_values('kelly_weights', ascending=False)
    return final