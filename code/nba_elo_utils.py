import os
import sys
import pandas as pd
import numpy as np
import math
import pytz
from datetime import datetime, timedelta
from geopy import distance

def silverK(MOV, elo_diff):
    """Calculate K constant (Source:
    https://www.ergosum.co/nate-silvers-nba-elo-algorithm/).

    Args:
        MOV - Margin of victory.
        elo_diff - ELO difference of teams.
    Returns:
        0, 1 - K constant
    """

    K_0 = 20
    if MOV > 0:
        multiplier = (MOV + 3) ** (0.8) / (7.5 + 0.006 * (elo_diff))
    else:
        multiplier = (-MOV + 3) ** (0.8) / (7.5 + 0.006 * (-elo_diff))

    return K_0 * multiplier, K_0 * multiplier


def silverS(home_score, away_score):
    """Calculate S for each team (Source:
    https://www.ergosum.co/nate-silvers-nba-elo-algorithm/).

    Args:
        home_score - score of home team.
        away_score - score of away team.
    Returns:
        0: - S for the home team.
        1: - S for the away team.
    """

    S_home, S_away = 0, 0
    if home_score > away_score:
        S_home = 1
    elif away_score > home_score:
        S_away = 1
    else:
        S_home, S_away = .5, .5

    return S_home, S_away


def elo_prediction(home_rating, away_rating):
    """Calculate the expected win probability of the home team (Source:
    https://www.ergosum.co/nate-silvers-nba-elo-algorithm/).

    Args:
        home_rating - initial home elo score.
        away_rating - initial away elo score.
    Returns:
        E_home - expected win probability of the home team.
    """

    E_home = 1. / (1 + 10 ** ((away_rating - home_rating) / (400.)))

    return E_home


def silver_elo_update(home_score, away_score, home_rating, away_rating):
    """Calculate change in elo for home and away teams Source:
    https://www.ergosum.co/nate-silvers-nba-elo-algorithm/).

    Args:
        home_score: score of home team.
        away_score: score of away team.
        home_rating: initial home elo score.
        away_rating: initial away elo score.
    Returns:
        0: change in home elo.
        1: change in away elo.
    """

    HOME_AD = 100.
    home_rating += HOME_AD
    E_home = elo_prediction(home_rating, away_rating)
    E_away = 1 - E_home
    elo_diff = home_rating - away_rating
    MOV = home_score - away_score

    S_home, S_away = silverS(home_score, away_score)
    if S_home > 0:
        K_home, K_away = silverK(MOV, elo_diff)
    else:
        K_home, K_away = silverK(MOV, elo_diff)

    return K_home * (S_home - E_home), K_away * (S_away - E_away)


def get_new_games():
    """ scrape espn to get new games for this season (until all star game)
        new_games - pandas dataframe of all new games.
    """

    seasons = pd.concat([pd.date_range("2018-10-16", "2019-02-14").to_series()])
    new_games = pd.DataFrame()  # Empty df to append new game data.
    replace_dict = {'WSH': 'WAS', 'NY': 'NYK', 'UTAH': 'UTA', 'SA': 'SAS', 'GS': 'GSW', 'NO': 'NOP'}

    # -- For each day since last game, check for game data.
    for day in pd.date_range(seasons[0], seasons[-1]):
        if day in seasons:
            print("Collecting data for games on: {}".format(day.date()))
            datestr = day.strftime("%Y%m%d")
            url = 'http://www.espn.com/nba/schedule/_/date/{0}'.format(datestr)
            df = pd.read_html(url)[0]
            #print(df.head())
            if ('result' in df.columns):
                results = df['result']
                print(results)

                for i in range(len(results)):
                    home_away = results[i].split(',')
                    team_id_home = home_away[0].split()[0]
                    pts_home = 0 #int(home_away[0].split()[1])
                    team_id_away = home_away[1].split()[0]
                    pts_away = 0 #int(home_away[1].split()[1])
                    new_games = new_games.append({'date_game': day,
                                                  'pts_away': pts_away,
                                                  'pts_home': pts_home,
                                                  'team_id_away': team_id_away,
                                                  'team_id_home': team_id_home}, ignore_index=True)

    new_games = new_games.replace({"team_id_away": replace_dict,
                                   "team_id_home": replace_dict})
    return new_games


def get_games_in_range(start, end):
    """ scrape espn to get new games for this season (until all star game)
        new_games - pandas dataframe of all new games.
    """

    seasons = pd.concat([pd.date_range(start, end).to_series()])
    new_games = pd.DataFrame()  # Empty df to append new game data.
    replace_dict = {'WSH': 'WAS', 'NY': 'NYK', 'UTAH': 'UTA', 'SA': 'SAS', 'GS': 'GSW', 'NO': 'NOP'}
    
    if seasons[0] >= datetime.strptime('2019-04-12','%Y-%m-%d'):
        new_format = True
        sep = 2
        home_idx = 1
        away_idx = 0
    else:
        new_format = False
        sep = 1
        home_idx = 0
        away_idx = 1
    
    # -- For each day since last game, check for game data.
    for day in pd.date_range(seasons[0], seasons[-1]):
        if day in seasons:
            print("Collecting data for games on: {}".format(day.date()))
            datestr = day.strftime("%Y%m%d")
            url = 'http://www.espn.com/nba/schedule/_/date/{0}'.format(datestr)
            df = pd.read_html(url)[0]
            #print(df.head())
            if ('result' in df.columns):
                results = df['result']
                #print(results)
                for i in range(0,len(results),sep):
                    home_away = results[i].split(',')
                    team_id_home = home_away[home_idx].split()[0]
                    pts_home = int(home_away[home_idx].split()[1])
                    team_id_away = home_away[away_idx].split()[0]
                    pts_away = int(home_away[away_idx].split()[1])
                    new_games = new_games.append({'date_game': day,
                                                  'pts_away': pts_away,
                                                  'pts_home': pts_home,
                                                  'team_id_away': team_id_away,
                                                  'team_id_home': team_id_home}, ignore_index=True)

    new_games = new_games.replace({"team_id_away": replace_dict,
                                   "team_id_home": replace_dict})
    return new_games

def last_elo(df):
    """Calculate the last ELO for each team (past 2016).
    Args:
        df - pandas dataframe containing database table.
    Returns:
        last_elo (dict) - dictionary where tm: elo score.
    """

    last_elo = {}
    df.date_game = pd.to_datetime(df.date_game)
    teams = df[df.date_game > pd.datetime(2016, 1, 1)].team_id_away.unique()

    for tm in teams:
        try:
            # -- Subset table to get most recent record with an ELO rating.
            tmp = df[((~df.elo_i_home.isnull()) & (df.team_id_away == tm)) |
                     ((~df.elo_i_home.isnull()) & (df.team_id_home == tm))] \
                .sort_values("date_game").iloc[-1]
        except:
            print("Error with: {}".format(tm))
        # -- Store ELO in dictionary.
        if tmp.team_id_home == tm:
            last_elo[tm] = tmp.elo_n_home
        else:
            last_elo[tm] = tmp.elo_n_away

    return last_elo


def new_season(elo_dict):
    """Update ELO score when rolling over into a new season.
    Args:
        elo_dict - last ELO scores for each team.
    Return:
        elo_dict - updated ELO scores.
    """

    for tm in elo_dict.keys():
        elo_dict[tm] = elo_dict[tm] * 0.75 + 1505 * 0.25

    return elo_dict


def update_elo(df, in_season=False):
    """Update ELO score for new records in nba_elo table and write to db.
    Args:
        df - pandas dataframe of nba_elo.
        elo - last ELO scores for each team.
        in_season - Do the ELO scores need to be rolled over between seasons?
    """

    seasons = [pd.date_range("2018-10-16", "2019-06-30").to_series()]

    df = df.copy()
    elo = last_elo(df)
    for season in seasons:
        if not in_season:
            elo = new_season(elo)

        for idx, row in df[df.date_game.isin(season)].iterrows():

            if (row.pts_home != None):
                home_tm, away_tm = row.team_id_home, row.team_id_away
                h_elo_i, a_elo_i = elo[home_tm], elo[away_tm]
                h_delta, a_delta = silver_elo_update(row.pts_home, row.pts_away, h_elo_i, a_elo_i)
                elo[home_tm] = h_elo_i + h_delta
                elo[away_tm] = a_elo_i + a_delta

                df.loc[idx, "elo_i_away"] = a_elo_i
                df.loc[idx, "elo_i_home"] = h_elo_i

                df.loc[idx, "elo_n_away"] = elo[away_tm]
                df.loc[idx, "elo_n_home"] = elo[home_tm]

    return df


def get_yesterdays_games():
    
    """ scrape espn to get just yesterdays games
        new_games - pandas dataframe of all new games.
    """

    day = pd.datetime.today() - timedelta(days=1)
    new_games = pd.DataFrame()  # Empty df to append new game data.
    replace_dict = {'WSH': 'WAS', 'NY': 'NYK', 'UTAH': 'UTA', 'SA': 'SAS', 'GS': 'GSW', 'NO': 'NOP'}

    datestr = day.strftime("%Y%m%d")
    url = 'http://www.espn.com/nba/schedule/_/date/{0}'.format(datestr)
    print("Collecting data for games on: {0}".format(datestr))
    
    df = pd.read_html(url)[0]

    if (df.shape[1] == 1):
        return -1
   
    if ('result' in df.columns):
        results = df['result']

        for i in range(len(results)):
            home_away = results[i].split(',')
            team_id_home = home_away[0].split()[0]
            pts_home = int(home_away[0].split()[1])
            team_id_away = home_away[1].split()[0]
            pts_away = int(home_away[1].split()[1])
            new_games = new_games.append({'date_game': day,
                                          'pts_away': pts_away,
                                          'pts_home': pts_home,
                                          'team_id_away': team_id_away,
                                          'team_id_home': team_id_home}, ignore_index=True)

        new_games = new_games.replace({"team_id_away": replace_dict,
                                       "team_id_home": replace_dict})
    return new_games


def get_todays_games(d):

    print("Getting games for {0}".format(d))
    if d is None:
        day = pd.datetime.today()
    else:
        day = datetime.strptime(d, '%Y-%m-%d')

    new_games = pd.DataFrame()  # Empty df to append new game data.
    replace_dict = {'WSH': 'WAS', 'NY': 'NYK', 'UTAH': 'UTA', 'SA': 'SAS', 'GS': 'GSW', 'NO': 'NOP'}

    datestr = day.strftime("%Y%m%d")
    url = 'http://www.espn.com/nba/schedule/_/date/{0}'.format(datestr)
    print(url)
    df = pd.read_html(url)[0]
    print(df.shape)
    
    if (df.shape[1] == 1):
        return new_games

    for i in range(len(df)):
        team_id_home = df['matchup'].iloc[i].split()[-1]
        pts_home = 0
        team_id_away = df['Unnamed: 1'].iloc[i].split()[-1]
        pts_away = 0
        new_games = new_games.append({'date_game': day,
                                      'pts_away': pts_away,
                                      'pts_home': pts_home,
                                      'team_id_away': team_id_away,
                                      'team_id_home': team_id_home}, ignore_index=True)

    new_games = new_games.replace({"team_id_away": replace_dict,
                                   "team_id_home": replace_dict})

    return new_games


def get_latest_elo(team):
    df = pd.read_csv('../data/nba_elo_19_latest.csv')

    row = df[(df.team_id_away == team) | (df.team_id_home == team)][-1:].iloc[0]
    if (row['team_id_away'] == team):
        elo = row['elo_n_away']
    else:
        elo = row['elo_n_home']

    return elo

def get_latest_elo_rollover(team):
    df = pd.read_csv('../data/nba_elo_19_latest.csv')

    row = df[(df.team_id_away == team) | (df.team_id_home == team)][-1:].iloc[0]
    if (row['team_id_away'] == team):
        elo = row['elo_n_away']*0.75 + 1505 * 0.25
    else:
        elo = row['elo_n_home']*0.75 + 1505 * 0.25

    return elo


def make_predictions(d=None):

    print("Getting predictions for {0}".format(d))
    next_games = get_todays_games(d)

    print(next_games.head())
    
    if (next_games.shape[0] == 0):
        print("No games scheduled")
        return

    for i in range(len(next_games)):

        home_team = next_games.iloc[i]['team_id_home']
        away_team = next_games.iloc[i]['team_id_away']

        print(home_team, away_team)

        home_elo = get_latest_elo_rollover(home_team)
        away_elo = get_latest_elo_rollover(away_team)

        elo_pred = elo_prediction(home_elo, away_elo)
        next_games.loc[i, 'home_win_prob'] = elo_pred

        mov = abs((home_elo + 100 - away_elo) / 28.)

        if (elo_pred > 0.5):
            pred_winner = home_team
        else:
            pred_winner = away_team

        next_games.loc[i, 'pred_winner'] = pred_winner
        next_games.loc[i, 'pred_mov'] = mov

    return next_games

# get geodesic distance between cities in miles
def get_distance(df, loc1, loc2):
    lat1 = df[df.code == loc1]['latitude'].iloc[0]
    lat2 = df[df.code == loc2]['latitude'].iloc[0]

    lon1 = df[df.code == loc1]['longitude'].iloc[0]
    lon2 = df[df.code == loc2]['longitude'].iloc[0]

    return distance.distance((lat1, lon1), (lat2, lon2)).miles


# get timezone offset between cities for a given timestamp
def get_tzoffset(df, loc1, loc2):
    tz1 = df[df.code == loc1]['tzrel'].iloc[0]
    tz2 = df[df.code == loc2]['tzrel'].iloc[0]

    return (tz1 - tz2)

def get_elevation(df, loc):
    return df[df.code == loc]['elevation'].iloc[0]

def days_since_last_game(dfn, team, day):
    last_day = dfn[(dfn.day_idx < day) & ((dfn.team_id_home == team) | (dfn.team_id_away == team))]['day_idx'].max()
    return day - last_day