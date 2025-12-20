import pandas as pd
from datetime import date, datetime, timezone, timedelta
from sportsreference.nba.schedule import Schedule
from sportsreference.nba.teams import Teams
from sportsreference.nba.boxscore import Boxscore
from sportsreference.nba.boxscore import Boxscores
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from basketball_reference_web_scraper import client
from sklearn import ensemble
from sklearn import linear_model
import os
import ast
from statsmodels.tools import add_constant
import statsmodels.api as sm
import unidecode
import re
import pytz
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer, quantile_transform


def utc_to_local(utc_dt):
    
    """utility to get datetime of games in a different time zone"""
    pacific = pytz.timezone('US/Pacific')
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(pacific)


def get_injured():
    
    """pulls the current injury data from rotowire and returns a dataframe of the injury players
    current injury is defined as out or out for the season. Additional status includes
    gametime decision"""
    
    os.system("wget -O injury.txt \"https://www.rotowire.com/basketball/tables/injury-report.php?Team=LAC&Pos=C\"")
    with open ("injury.txt", "r") as myfile:
        data=myfile.read().replace('\n', '')
    data = data[1:-1]
    mydict = ast.literal_eval(data)
    df = pd.DataFrame.from_dict(mydict)
    dff = df[df.status.isin(['Out','Out For Season'])].copy()
    dff.rename(columns={'team':'abb','player':'name'},inplace=True)
    
    #roto-wire has some real special abbreviations for team names
    abb_mappings  = {'CHA':'CHO', 'NY': 'NYK', 'NO':
       'NOP', 'BKN':'BRK', 'GS': 'GSW','SAN':'SAS'}

    dff.abb = dff.abb.replace(abb_mappings)
    #this doesn't work
    dff['player_id'] = dff.lastname.map(lambda x: x[:6].lower()) + dff['firstname'].map(lambda x: x[:2].lower()) + '01'
    return dff[['name','abb','player_id']] #,df



def roster_minutes_injuries(team, dfp, dfr, num_games):
    
    """this gets a baseline minutes projection and joins it with raptor ratings
    the minutes projection is based on the number of games used. 
    by default it also looks up if a player is marked as injured
    
    used to help predict an arbitrary match up 
    
    arg: 
    
    dfp: a data set that contains each players id and minutes played that seasons
    dfr: a dataframe that contains each players 
    num_games: the lookback for the number of games to include in compute the 
    average minutes per game. the project for minutes is the average of the last
    few gamges plus adjustments
    
    returns:
    
    dft a data with each players projected seconds played taking into account
    injuries
    """
    
    games = dfp[dfp.team == team].game_index.unique()
    games = np.sort(games)
    last = games[-num_games:]

    df_temp = dfp[(dfp.team == team) & (dfp.game_index.isin(last))]

    seconds_projection = (df_temp.groupby('slug')[['act_seconds_played','seconds_played_adj']].sum()).reset_index()
    
    seconds_projection.rename(columns=({'act_seconds_played':'seconds_played'}), inplace=True)
    
    dft = pd.merge(seconds_projection, dfr, left_on = 'slug', right_on = 'player_id')
    
    dft['seconds_played'] = dft['seconds_played']/float(num_games)
    dft['ratings_product_offense'] = dft['raptor_offense']*dft['seconds_played']
    dft['ratings_product_defense'] = dft['raptor_defense']*dft['seconds_played']
    dft['ratings_product_pace'] = dft['pace_impact']*dft['seconds_played']
    
    dft['team'] = team
    
    #df_dc = pd.read_csv('data/team_depth_charts.csv').iloc[:,1:]
    df_dc = pd.read_csv('../data/team_depth_charts.csv').iloc[:,1:]
    df_dc.name = df_dc.name.map(lambda x: unidecode.unidecode(x))

    dftm = pd.merge(df_dc, dft, left_on = 'name', right_on = 'player_name', how='right')

    #if missing position, make them SF
    dftm.position.fillna(2.0, inplace=True)

    players_to_zap = dftm[dftm.seconds_played_adj == 0].player_name.unique()
    if len(players_to_zap) > 0:
        for player in players_to_zap:
            dftm = zap_player_pos(dftm, player, .85)
        
    return dftm

def game_agg(dft_home, dft_away, df_sup_current, raptor_slope):

    """computes the features needed by the model to predict the score 
    and win probabilities
    
    args: 
    
    dft_home: the home team's roster with minutes and raptor ratings
    dft_away: the away team's roster with minutes and raptor ratings
    
    returns:
    
    """
    
    home_team = dft_home['team'].iloc[0]
    dft_home['location'] = 'HOME'
    
    dft_away['location'] = 'AWAY'
    away_team = dft_away['team'].iloc[0]

    
    df_teams = dft_home.append(dft_away, ignore_index=True)
    
    cols = ['ratings_product_offense','ratings_product_defense','ratings_product_pace','seconds_played']
    
    dft_agg = df_teams.groupby(['location','team'])[cols].sum().reset_index()
    dft_agg['game_desc'] =  home_team + '_v_' + away_team
    
    dfg = game_raptor_calcs(dft_agg, df_sup_current, raptor_slope)
    
    return dfg

def game_raptor_calcs(dft, df_sup_current, raptor_slope):
    
    """compute raptor featuers needed for model"""
    
    
    PLAYERS = 5
    h = np.where(dft.location == 'HOME', 1, 0)
    a = 1-h
    
    d = df_sup_current[['pace_mean','home_rate_mean','away_rate_mean']].mean().to_dict()
    dft['home_rate_mean'] = d['home_rate_mean']
    dft['away_rate_mean'] = d['away_rate_mean']
    dft['pace_mean'] = d['pace_mean']
   
    o_mean = h*dft['home_rate_mean'] + a*dft['away_rate_mean']
    d_mean = a*dft['home_rate_mean'] + h*dft['away_rate_mean']
    dft['ORT'] = raptor_slope * PLAYERS * (dft['ratings_product_offense']/dft['seconds_played'])
    dft['DRT'] = .95 * PLAYERS * (dft['ratings_product_defense']/dft['seconds_played'])
    dft['pace'] = .95 * PLAYERS * (dft['ratings_product_pace']/dft['seconds_played'])
    dft['p_scored'] = ((o_mean + dft['ORT']) * (dft['pace'] + dft['pace_mean'])/100.)
    dft['p_allowed'] = ((d_mean - dft['DRT']) * (dft['pace'] + dft['pace_mean'])/100.)
    
    #reshape to one row per game
    game_data = dft.reset_index().set_index(['game_desc','location'])[['p_allowed','p_scored','team',
                                                                       'pace','ORT','DRT']].unstack()
    
    #format the reshaped data
    data = game_data.values
    vals = game_data.columns.levels[0]
    home_away = game_data.columns.levels[1]
    col_names = [loc + '_' + val for val in vals for loc in home_away]

    dfg = pd.DataFrame(data = game_data.values, columns = col_names, index = game_data.index)
    
    return dfg



def make_today_features(dfp_adj, dfr_current, df_sup_current, raptor_slope = 0.78, days=6):
    
    today = datetime.now(pytz.timezone('US/Pacific'))
    month = today.month
    year = today.year
    day = today.day
    season_year = year if month < 9 else year + 1
    season_obj = client.season_schedule(season_end_year=season_year)

    todays_games = [g for g in season_obj if utc_to_local(g['start_time']).date()
                    == datetime(year,month, day).date()]
    
    df_today = pd.DataFrame()
    df_rosters = pd.DataFrame()
    
    for game in todays_games:
        away_team = game['away_team'].value
        home_team = game['home_team'].value
    
        dft_away = roster_minutes_injuries(away_team, dfp_adj, dfr_current, days)
        df_rosters = df_rosters.append(dft_away, sort=False)

        dft_home = roster_minutes_injuries(home_team, dfp_adj, dfr_current, days)
        df_rosters = df_rosters.append(dft_home, sort=False)

        df_game = game_agg(dft_home, dft_away, df_sup_current, raptor_slope)
        df_today = df_today.append(df_game,sort=False)
        
    d = get_abbreviation_mapping()
    df_today.reset_index(inplace=True)
    index_date = today.date().isoformat().replace('-','')
    print(df_today.columns)
    df_today['game_index'] =  index_date + '0' + df_today['HOME_team'].map(d)
    df_today.set_index('game_index',inplace=True)

    df_today['combined_pace'] = df_today.AWAY_pace + df_today.HOME_pace
    df_today['abs_eff_diff'] = np.abs((df_today['AWAY_ORT'] - df_today['AWAY_DRT'])
                                 - (df_today['HOME_ORT'] - df_today['HOME_DRT']))

    return df_today, df_rosters

def get_team_list(start_year, end_year):
    
    """get all teams over an interval and return a dictionary of dataframes 
    with the teams in each season"""
    
    number_of_years = end_year - start_year + 1
    
    teams = {start_year+i: Teams(start_year+i) for i in range(number_of_years)} 

    return {year: obj.dataframes  for year, obj in teams.items()}

def get_games(df_teams):
    
    for year in df_teams.keys():
        
        dfs = pd.DataFrame()
            
        team_list = df_teams[year].index.values
        for team in team_list:
            try: 
                team_schedule = Schedule(team, year)
                df_temp = team_schedule.dataframe_extended
                df_temp['team'] = team
                dfs = dfs.append(df_temp)
            except(ValueError):
                pass
            
        dfs['year'] = year
        dfs.to_csv('../data/{}_season_data.csv'.format(year))
        dfs.to_pickle('../data/{}_season_data.pkl'.format(year))
        print ('year {} done'.format(year))
    return dfs

def train_models(df_train,points_features,prob_features):
    today = pd.Timestamp(datetime.now(pytz.timezone('US/Pacific')).date())
    weekday = today.isoweekday()
    weekday = weekday-1 if weekday > 0 else 6

    df_train['dayofweek'] = df_train.date.map(lambda x: x.dayofweek)
    df_train['month'] = df_train.date.map(lambda x: x.month)
    #team_cols = [col for col in df_train.columns if ((col[0] == 'a') | (col[0] == 'h'))]
    #month_cols = [col for col in df_train.columns if (col[0] == 'M_') ]

    df_train['weight'] = 1.0

    df_train['weight'] = (df_train.dayofweek == weekday) * 1.5 + df_train['weight']
    df_train['weight'] = (df_train.month == today.month) * 1.0 + df_train['weight']

    points_model = TransformedTargetRegressor(
        regressor=Ridge(alpha=25.0),
        transformer=QuantileTransformer(n_quantiles=200,
                                        output_distribution='normal'))
   

    points_model.fit(df_train[points_features], df_train[['AWAY_points','HOME_points']], sample_weight = df_train['weight'])
    df_train, prob_model = train_probs_lr(df_train, points_model, points_features, prob_features)
    
    return df_train, points_model, prob_model

def raptor_features_season(dfp, dfr, ratings_slope, avg_otr, avg_pace):
    
    
    """function to create raptor features for a season or slate of games
    takes the raptor ratings and mintues to create per team, game raptor ratings
    
    dfp: a data set that contains each players id and minutes played that seasons
    dfr: a dataframe that contains each players 
    
    ratings_slope: scale the impact of agg raptor ratings to projected avg OTR
    avg_otr: the season average otr
    avg_pace: the seasons average_pace
    
    """
    
    dfm = pd.merge(dfp, dfr, left_on = 'slug', right_on = 'player_id')

    dfm.rename(columns = {'Unnamed: 0': 'game_index'}, inplace=True)
    dfm['ratings_product_offense'] = dfm['raptor_offense']*dfm['seconds_played']
    dfm['ratings_product_defense'] = dfm['raptor_defense']*dfm['seconds_played']
    dfm['ratings_product_pace'] = dfm['pace_impact']*dfm['seconds_played']
    dfm['points'] = ((dfm['made_field_goals']-dfm['made_three_point_field_goals'])*2
                     + dfm['made_free_throws'] + 3*dfm['made_three_point_field_goals'])
    dft = dfm.groupby(['game_index','location','team'])[['points','ratings_product_offense',
                                                     'ratings_product_defense','ratings_product_pace',
                                                         'seconds_played']].sum()
    PLAYERS = 5
    ADJ = ratings_slope
    AVG_ORT = avg_otr
    AVG_PACE = avg_pace
    dft['ORT'] = ADJ * PLAYERS * (dft['ratings_product_offense']/dft['seconds_played'])
    dft['DRT'] = ADJ * PLAYERS * (dft['ratings_product_defense']/dft['seconds_played'])
    dft['pace'] = ADJ*PLAYERS*(dft['ratings_product_pace']/dft['seconds_played'])
    dft['p_scored'] = (AVG_ORT + dft['ORT']) * (dft['pace'] + AVG_PACE)/100. 
    dft['p_allowed'] = (AVG_ORT - dft['DRT']) * (dft['pace'] + AVG_PACE)/100.

    #reshape to one row per game

    game_data = dft.reset_index().set_index(['game_index','location'])[['p_allowed','p_scored',
                                                                        'team','points','ORT','DRT','pace']].unstack()
    data = game_data.values
    vals = game_data.columns.levels[0]
    home_away = game_data.columns.levels[1]
    col_names = [loc + '_' + val for val in vals for loc in home_away]

    dfg = pd.DataFrame(data = game_data.values, columns = col_names, index = game_data.index)
    
    return dfg

def raptor_features_season_v2(dfp, dfr, df_sup, ratings_slope):
    
    
    """function to create raptor features for a season or slate of games
    takes the raptor ratings and mintues to create per team, game raptor ratings
    
    dfp: a data set that contains each players id and minutes played that seasons
    dfr: a dataframe that contains each players 
    
    ratings_slope: scale the impact of agg raptor ratings to projected avg OTR
    avg_otr: the season average otr
    avg_pace: the seasons average_pace
    
    """
    dfp['player_id'] = dfp['slug']
    #print (dfp['player_id'].nunique())
    dfm = pd.merge(dfp, dfr, on = ['player_id','season'], suffixes = ('','_r'))
    #print (dfm['player_id'].nunique())
    
    #print(dfm.columns)
    #print(dfm.head())

    dfm.rename(columns = {'Unnamed: 0': 'game_index'}, inplace=True)
    dfm['ratings_product_offense'] = dfm['raptor_offense']*dfm['seconds_played']
    dfm['ratings_product_defense'] = dfm['raptor_defense']*dfm['seconds_played']
    dfm['ratings_product_pace'] = dfm['pace_impact']*dfm['seconds_played']
    dfm['points'] = ((dfm['made_field_goals']-dfm['made_three_point_field_goals'])*2
                     + dfm['made_free_throws'] + 3*dfm['made_three_point_field_goals'])
    dft = dfm.groupby(['game_index','location','team'])[['points','ratings_product_offense',
                                                     'ratings_product_defense','ratings_product_pace',
                                                         'seconds_played']].sum()
    dft.reset_index(inplace=True)
    print(dft.shape)
    dft = pd.merge(dft, df_sup, on = 'game_index')
    print(dft.shape)
    PLAYERS = 5
    ADJ = ratings_slope
    #print (dft.head())
    h = np.where(dft.location == 'HOME', 1, 0)
    a = 1-h
   
    o_mean = h*dft['home_rate_mean'] + a*dft['away_rate_mean']
    d_mean = a*dft['home_rate_mean'] + h*dft['away_rate_mean']
    dft['ORT'] = ADJ * PLAYERS * (dft['ratings_product_offense']/dft['seconds_played'])
    dft['DRT'] = .95 * PLAYERS * (dft['ratings_product_defense']/dft['seconds_played'])
    dft['pace'] = .95 * PLAYERS * (dft['ratings_product_pace']/dft['seconds_played'])
    dft['p_scored'] = ((o_mean + dft['ORT']) * (dft['pace'] + dft['pace_mean'])/100.)
    dft['p_allowed'] = ((d_mean - dft['DRT']) * (dft['pace'] + dft['pace_mean'])/100.)
    dft['points'] = dft['points']*dft['score_norm']

    #reshape to one row per game

    game_data = dft.set_index(['game_index','location'])[['p_allowed','p_scored',
                                                                        'team','points','ORT','DRT','pace']].unstack()
    data = game_data.values
    vals = game_data.columns.levels[0]
    home_away = game_data.columns.levels[1]
    col_names = [loc + '_' + val for val in vals for loc in home_away]

    dfg = pd.DataFrame(data = game_data.values, columns = col_names, index = game_data.index)

    dfg['combined_pace'] = dfg.AWAY_pace + dfg.HOME_pace
    dfg['abs_eff_diff'] = np.abs((dfg['AWAY_ORT'] - dfg['AWAY_DRT'])
                                 - (dfg['HOME_ORT'] - dfg['HOME_DRT'])) 

    return dfg

def evaluate_model_current(model, df_train, df_current, features):
    
    
    """evaluates the model out of sample on the current season"""
    #features = ['AWAY_p_allowed', 'HOME_p_allowed', 'AWAY_p_scored', 'HOME_p_scored']
    
    model.fit(df_train[features], df_train[['AWAY_points','HOME_points']])
    
    #print (model.coef_)
    #print (model.intercept_)
    
    y_pred_current = model.predict(df_current[features])
    
    mae = np.abs(y_pred_current - df_current[['AWAY_points','HOME_points']]).mean()
        
    accuracy = np.where((y_pred_current[:,0] > y_pred_current[:,1]) == 
                 (df_current['AWAY_points']>df_current['HOME_points']), 1, 0).mean()
        
    r = {'MAE_away': mae['AWAY_points'],
                        'MAE_home' : mae['HOME_points'],
                        'Accuracy': accuracy}
    
    return pd.Series(r), y_pred_current


def zap_player_simple(dft, player_name):
    
    """helper function to reallocate a players mintues through the rest of the team
    simple because it does it across the entire roster rather than per position"""
    
    idx = dft[dft.player_name == player_name].index[0]
    seconds_to_allocate = dft.loc[idx,'seconds_played'] 

    dff = dft[dft.player_name != player_name].copy()
    
    T = 60.
    score = (1300 - np.abs(dff.seconds_played - 1200))/T
    dff['seconds_played'] = dff['seconds_played'] + seconds_to_allocate * (np.exp(score)/np.exp(score).sum())

    dff['ratings_product_offense'] = dff['raptor_offense']*dff['seconds_played']
    dff['ratings_product_defense'] = dff['raptor_defense']*dff['seconds_played']
    dff['ratings_product_pace'] = dff['pace_impact']*dff['seconds_played']
    
    return dff

def set_minutes_simple(dft, player_name, minutes):
    
    """helper function to set a players minutes to a specific number
    simple because it does it across the entire roster rather than per position"""
    
    idx = dft[dft.player_name == player_name].index[0]
    current_seconds = dft.loc[idx,'seconds_played']
    seconds_to_allocate = current_seconds - 60*minutes

    i = np.where(dft.player_name != player_name, 1, 0)
    
    T = 60.
    score = np.exp((1300 - np.abs(dft.seconds_played - 1200))/T)
    score = i * score
    dft['seconds_played'] = dft['seconds_played'] + seconds_to_allocate * score/score.sum()

    dft['ratings_product_offense'] = dft['raptor_offense']*dft['seconds_played']
    dft['ratings_product_defense'] = dft['raptor_defense']*dft['seconds_played']
    dft['ratings_product_pace'] = dft['pace_impact']*dft['seconds_played']
    
    return dft

def regularize_raptor(dfr, season, prior_weight):
    
    """smoothes raptor ratings to the replacement level mean
    
    args: 
    
    dfr: the raptor weights
    prior_weight: how much to smooth to the mean"""
    #dfrp = pd.read_csv('data/raptors_player_stats.csv')
    dfrp = pd.read_csv('../data/raptors_player_stats.csv')
    dfrp = dfrp[dfrp.season==(season-1)].copy()
    dfrm = pd.merge(dfr, dfrp, how='left', on = 'player_id', suffixes=('', '_p'))
    dfrm.mp_p.fillna(0, inplace=True)
    dfrm.raptor_offense_p.fillna(0, inplace=True)
    dfrm.raptor_defense_p.fillna(0, inplace=True)
    dfrm.pace_impact_p.fillna(0, inplace=True)
    off_pw = (300./(300.+dfrm['mp'].astype(float)))
    def_pw = (100./(100.+dfrm['mp'].astype(float)))

    
    dfrm['raptor_offense'] = dfrm['raptor_offense'] * (1-off_pw) + dfrm['raptor_offense_p'] * (off_pw)
    dfrm['raptor_defense'] = dfrm['raptor_defense'] * (1-def_pw) + dfrm['raptor_defense_p'] * (def_pw)
    dfrm['pace_impact'] = dfrm['pace_impact'] * (1-def_pw) + dfrm['pace_impact_p'] * (def_pw)

    
    lower_quantile = dfrm.mp.quantile(q=.25)
    
    df_scrubs = dfrm[dfrm.mp < lower_quantile ] 
    
    repl_off = (df_scrubs.mp * df_scrubs.raptor_offense).sum()/(df_scrubs.mp.sum())
    repl_def = (df_scrubs.mp * df_scrubs.raptor_defense).sum()/(df_scrubs.mp.sum())
    repl_pace = (df_scrubs.mp * df_scrubs.pace_impact).sum()/(df_scrubs.mp.sum())
    
    sw = (prior_weight/(prior_weight+dfrm['mp'].astype(float) + .6*dfrm['mp_p']))

    dfrm['raptor_offense'] = ((dfrm['raptor_offense'] *(1-sw))  
                                         + (repl_off * sw))
    dfrm['raptor_defense'] = ((dfrm['raptor_defense'] *(1-sw))  
                                         + (repl_def * sw))
    dfrm['pace_impact'] = ((dfrm['pace_impact'] *(1-sw))  
                                      + (repl_pace * sw))
    
    return dfrm


def get_abbreviation_mapping():
    
    """help function to map between different team names"""
    #df_season = pd.read_csv('data/2020_season_data.csv')
    df_season = pd.read_csv('../data/2020_season_data.csv')
    d = dict()
    for i in df_season.groupby(['winning_abbr','winning_name'])['away_assists'].count().keys():
        value = i[0]
        key = i[1].upper()
        d[key] = value
        
    return d

def get_last_player_minutes():
    
    """pulls data from basketball reference to get the most recent player minutes"""
    
    d = get_abbreviation_mapping()
    
    today = datetime.now().date()
    #print(today)
    year = 2021
    season = client.season_schedule(season_end_year=year)
    dates = []
    for g in season:
        date = g['start_time'].date()
        dates.append(date)
    dates = list(set(dates))
    dates = [date for date in dates if date < today]
    last = [max(dates)]
    
    
    dfl = pd.DataFrame()
    
    for date in last:
    
        index_date = date.isoformat().replace('-','')
        player_boxscores = client.player_box_scores(day=date.day, month=date.month, year=date.year)
        
        for row in player_boxscores:
    
            loc = row['location'].value
            
            if loc == 'AWAY':
                index_team = d[row['opponent'].value]
            else:
                index_team = d[row['team'].value]
            
            game_index = index_date + '0' + index_team
            s = pd.Series(row)
            s.name = game_index
            dfl = dfl.append(s)
        
    dfl.opponent = dfl.opponent.map(lambda x: x.value)
    dfl.team = dfl.team.map(lambda x: x.value)
    dfl.location = dfl.location.map(lambda x: x.value)
    dfl.outcome = dfl.outcome.map(lambda x: x.value)
    dfl['player_key'] = dfl.location.map(lambda x: str(x)) + dfl.slug.map(lambda x: str(x) )
    
    return dfl

def get_season_game_dates(season):
    """
    :param season:
    :return: sorted list of game dates for a season
    """

    today = datetime.now().date()
    season = client.season_schedule(season_end_year=season)
    dates = []
    for g in season:
        date = g['start_time'].date()
        dates.append(date)
    dates = list(set(dates))
    dates = [date for date in dates if date < today]
    return sorted(dates)

def make_sup_current(dfe):
    
    """function is used to create efficiency and pace season averages
    and also get adjustment for OT games """
    dfs = dfe[['away_points','home_points','away_offensive_rating','home_offensive_rating','pace']].astype(float)
    dfs['date'] = dfe['date']
    dfs['game_index'] = dfe['game_index']
    dfs['poss'] = 100*dfs['away_points']/dfs['away_offensive_rating']
    dfs['est_minutes'] = dfs['poss']*48/dfs['pace']
    dfs['score_norm'] = np.round(48/dfs['est_minutes'],2)
    year = dfs['game_index'].map(lambda x: x[:4])
    month = dfs['game_index'].map(lambda x: x[4:6])
    day = dfs['game_index'].map(lambda x: x[6:8])
    dfs['date'] = pd.to_datetime(year + '-' + month + '-' + day)
    # removes playoffs and last game of season
    # dfs = dfs[dfs.date < '2020-04-10']
    means = dfs[['pace','home_offensive_rating','away_offensive_rating']].mean()
    dfs['pace_mean'] = means['pace']
    dfs['home_rate_mean'] = means['home_offensive_rating']
    dfs['away_rate_mean'] = means['away_offensive_rating']
    return dfs[['game_index','score_norm','pace_mean','home_rate_mean','away_rate_mean']]

def get_latest_effiency_data(dates):
    
    """pulls data from basketball reference to get recent efficiency data"""
    
    df_cached = pd.read_csv('../data/latest_game_eff.csv')
    df_cached = df_cached.iloc[:,1:]

    last_date_str = df_cached.date.max()
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()

    dfe = pd.DataFrame()
    dates_to_pull = [date for date in dates if date > last_date]
    for date in dates_to_pull:
        team_boxscores = Boxscores(date)
        for game in list(team_boxscores.games.values())[0]:
            team = game['home_abbr']
            game_date = date.isoformat()
            row = get_game_stats(team, game_date)
            dfe = dfe.append(row)
        
    dfe = df_cached.append(dfe)
    dfe.to_csv('../data/latest_game_eff.csv')
    
    return dfe

def get_recent_player_minutes():
    
    """pulls data from basketball reference to get recent playing data"""
    
    d = get_abbreviation_mapping()
    dates = get_season_game_dates(2021)
    df_cached = pd.read_csv('../data/recent_player_minutes.csv')
    #df_cached = pd.read_csv('data/recent_player_minutes.csv')
    #print(len(df_cached))
    #print(df_cached.columns)
    last_date_str = df_cached.game_index.values[-1][:-4]
    last_date = datetime.strptime(last_date_str, '%Y%m%d').date()

    dfl = pd.DataFrame()
    
    for date in dates:

        if (date > last_date):
            index_date = date.isoformat().replace('-','')
            player_boxscores = client.player_box_scores(day=date.day, month=date.month, year=date.year)

            for row in player_boxscores:

                loc = row['location'].value

                if loc == 'AWAY':
                    index_team = d[row['opponent'].value]
                else:
                    index_team = d[row['team'].value]

                game_index = index_date + '0' + index_team
                s = pd.Series(row)
                s.name = game_index
                dfl = dfl.append(s)

    if len(dfl) > 0:
        dfl.opponent = dfl.opponent.map(lambda x: x.value)
        dfl.team = dfl.team.map(lambda x: x.value)
        dfl.location = dfl.location.map(lambda x: x.value)
        dfl.outcome = dfl.outcome.map(lambda x: x.value)
        dfl['player_key'] = dfl.location.map(lambda x: str(x)) + dfl.slug.map(lambda x: str(x) )
        dfl = dfl.reset_index()
        dfl = dfl.rename(columns={'index': 'game_index'})

        df_cached = df_cached.append(dfl)
        df_cached.to_csv('../data/recent_player_minutes.csv', index=None)
        #df_cached.to_csv('data/recent_player_minutes.csv', index=None)
        
    return df_cached, dates


def get_latest_raptor():
    
    """pull raptor ratings from 538 and save"""
    
    today = datetime.now().date()
    year = today.year
    month = today.month
    day = today.day
    df = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/2021/latest_RAPTOR_by_team.csv')
    sday = today.strftime("%d")
    smonth = today.strftime("%m")
    df.to_csv('latest_RAPTOR_by_team_{0}-{1}-{2}.csv'.format(year,smonth,sday))
    
    df = df.drop_duplicates('player_id')
    
    return df

def predict_scores(dfg, score_model, prob_model, features):
    
    """predicts scores for a given game and formats the dsata"""
    
    away = dfg['AWAY_team'].values[0]
    home = dfg['HOME_team'].values[0]

    p_score = score_model.predict(dfg[features])
    away_p = p_score[0,0]
    home_p = p_score[0,1]
    ratio = home_p/(home_p+away_p)
    
    f = np.array([1, ratio, home_p, away_p])
    
    p_prob = prob_model.predict(f)[0]
    
    r = {'home': [home, round(home_p,1), round(1-p_prob,3)],
    'away': [away, round(away_p,1) , round(p_prob,3)]}
    
    return pd.DataFrame(r, index=['team','predicted score','win_prob'])

def get_historical_data(season_list):
    
    dfr = pd.read_csv('../data/raptors_player_stats.csv')
    #dfr = pd.read_csv('data/raptors_player_stats.csv')
    dfr = dfr[dfr.season.isin(season_list)]
    dfp = pd.DataFrame()
    for season in season_list:
        df_temp = pd.read_csv('../data/{}_player_boxscores.csv'.format(season))
        #df_temp = pd.read_csv('data/{}_player_boxscores.csv'.format(season))
        df_temp['season'] = season
        dfp = dfp.append(df_temp)
    return dfr, dfp

def rolling_avg(group):
    
    return group.act_seconds_played.rolling(6, min_periods=1).mean().shift(1).fillna(method='bfill')


def add_dummies(dfr_games):
    
    dfr_games.reset_index(inplace=True)
    dfr_games = dfr_games.join(pd.get_dummies(dfr_games.game_index.map(lambda x: x[4:6]), 
                                              drop_first=True, prefix='M'))
    dfr_games = dfr_games.join(pd.get_dummies(dfr_games.AWAY_team, drop_first=True, prefix='a'))
    dfr_games = dfr_games.join(pd.get_dummies(dfr_games.HOME_team, drop_first=True, prefix='h'))
    
    dfr_games['date'] = pd.to_datetime(dfr_games.game_index.map(lambda x: x[:4])+ '-' + 
                                 dfr_games.game_index.map(lambda x: x[4:6])+'-'
                                 + dfr_games.game_index.map(lambda x: x[6:8]))
    return dfr_games

def train_probs_lr(df_train, points_model, points_features, prob_features):
    
    y_pred = points_model.predict(df_train[points_features])
    df_train['model_home_p'] = y_pred[:,1]
    df_train['model_away_p'] = y_pred[:,0]
    df_train['model_total_p'] = df_train['model_home_p']+df_train['model_away_p']
    df_train['total'] = df_train['AWAY_points']+df_train['HOME_points']
    df_train['diff'] = df_train['model_away_p']-df_train['model_home_p']
    df_train['diff_x_total'] = df_train['diff']*df_train['model_total_p']

    df_train['away_win'] = np.where(df_train['AWAY_points'] > df_train['HOME_points'],1,0)
    
    lr = linear_model.LogisticRegression(C=100.,solver='liblinear', intercept_scaling =100)
    lr.fit(df_train[prob_features],df_train['away_win'])
    df_train['away_win_p'] = lr.predict_proba(df_train[prob_features])[:,1]
    
    return df_train, lr

def project_seconds_played(dfp):
    
    dfp.rename(columns = ({'Unnamed: 0': 'game_index'}), inplace=True)
    dfp.sort_values(['slug','game_index'], inplace=True)
    dfp['seconds_played'] = dfp.groupby(['slug']).apply(rolling_avg).values
    return dfp

def pred_points(df_test, points_model, features):
    
    y_pred = points_model.predict(df_test[features])
    df_test['model_home_p'] = y_pred[:,1]
    df_test['model_away_p'] = y_pred[:,0]
    df_test['model_total_p'] = df_test['model_home_p']+df_test['model_away_p']
    df_test['total'] = df_test['AWAY_points']+df_test['HOME_points']
    df_test['diff'] = df_test['model_away_p']-df_test['model_home_p']
    df_test['diff_x_total'] = df_test['diff']*df_test['model_total_p']

    df_test['away_win'] = np.where(df_test['AWAY_points'] > df_test['HOME_points'],1,0)
    
    return df_test

def pred_probs(df_test, prob_model, prob_features):
     
    df_test['away_win_p'] = prob_model.predict_proba(df_test[prob_features])[:,1]
    
    return df_test

def get_sup_data(season_list):
    """gets pacing and other adjustment data"""
    df_sup = pd.DataFrame()
    for season in season_list:
        #df_sup = df_sup.append(pd.read_csv('data/{}_reg_sup.csv'.format(season)))
        df_sup = df_sup.append(pd.read_csv('../data/{}_reg_sup.csv'.format(season)))
    
    return df_sup

def train_historical(dfp, dfr, raptor_slope, avg_otr, avg_pace, model, features):
    
    dfr_games = raptor_features_season(dfp, dfr, .8, 108., 98.91)
    
    model.fit(dfr_games[features], dfr_games[['AWAY_points','HOME_points']])
    
    return model, dfr_games


def current_model_calibration(model, df_games_train, df_curr_games, features):
    
    """gets the win probabilities and displays info about game"""
    
    acc, currrent_predict = evaluate_model_current(model, df_games_train, df_curr_games, features)
    df_curr_games['model_home_p'] = currrent_predict[:,1]
    df_curr_games['model_away_p'] = currrent_predict[:,0]
    df_curr_games['model_total_p'] = df_curr_games['model_home_p']+df_curr_games['model_away_p']
    df_curr_games['total'] = df_curr_games['AWAY_points']+df_curr_games['HOME_points']
    #(df_curr_games['total']-df_curr_games['model_total_p']).rolling(200).mean().plot()
    
    df_curr_games['ratio'] =  df_curr_games['model_home_p']/(df_curr_games['model_home_p']+df_curr_games['model_away_p'])
    df_curr_games['away_win'] = np.where(df_curr_games['AWAY_points'] > df_curr_games['HOME_points'],1,0)
    x = add_constant(df_curr_games[['ratio','model_home_p','model_away_p']])
    logit = sm.Logit(df_curr_games['away_win'], x)
    prob_model = logit.fit()
    #print (np.where((prob_model.predict(x) > .5) == (df_curr_games['away_win']==1), 1, 0).mean())
    
    return df_curr_games, prob_model

def get_inj_adjusted_minutes(dfp):
    
    d = get_abbreviation_mapping()
    df_inj = get_injured()
    dfp['abb'] = dfp['team'].map(d)
    dfp.name = dfp.name.map(lambda x: unidecode.unidecode(x))
    dfp_ci = pd.merge(dfp, df_inj[['player_id','name','abb']],
                           how = 'left', on = ['name','abb'], suffixes=('_m',''))
    dfp_ci['seconds_played_adj'] = dfp_ci['seconds_played'] * dfp_ci.player_id.isnull().astype(int)
    
    return dfp_ci
    

def get_current_data():

    dfr_current = get_latest_raptor()
    dfp_current, dates = get_recent_player_minutes()
    dfp_current['season'] = 2021
    dfp_current['act_seconds_played'] = dfp_current['seconds_played']
    
    return dfr_current, dfp_current, dates 

def predict_todays_games(dfp_adj, dfr_current, model, prob_model, features,
                         raptor_slope, avg_ort, avg_pace, days=3):
    
    #get data needed to predict today's games
    today = datetime.now(pytz.timezone('US/Pacific'))
    month = today.month
    year = today.year
    day = today.day
    season_year = year if month < 9 else year + 1
    season_obj = client.season_schedule(season_end_year=season_year)

    todays_games = [g for g in season_obj if utc_to_local(g['start_time']).date()
                    == datetime(year,month, day).date()]
    #print(todays_games)
    
    df_games_csv = pd.DataFrame()
    df_games_display = pd.DataFrame()
    df_rosters = pd.DataFrame()
    for game in todays_games:
        away_team = game['away_team'].value
        home_team = game['home_team'].value
    
        dft_away = roster_minutes_injuries(away_team, dfp_adj, dfr_current, days)
        df_rosters = df_rosters.append(dft_away, sort=False)

        dft_home = roster_minutes_injuries(home_team, dfp_adj, dfr_current, days)
        df_rosters = df_rosters.append(dft_home, sort=False)

        dfg = game_agg(dft_home, dft_away, raptor_slope, avg_ort, avg_pace)
        p = predict_scores(dfg, model, prob_model, features)
        df_games_display = df_games_display.append(p)
        #print (p)
        df_games_csv = df_games_csv.append(dfg)

    y_pred = model.predict(df_games_csv[features])
    df_games_csv['AWAY_points_p'] = y_pred[:,0]
    df_games_csv['HOME_points_p'] = y_pred[:,1]
    df_games_csv['HOME_ratio'] = df_games_csv['HOME_points_p']/(df_games_csv['HOME_points_p']+df_games_csv['AWAY_points_p'])
    x = add_constant(df_games_csv[['HOME_ratio','HOME_points_p','AWAY_points_p']])
    df_games_csv['HOME_wins_p'] = 1-prob_model.predict(x)
    
    sday = today.strftime("%d")
    smonth = today.strftime("%m")
    now = datetime.now()
    #hour = now.strftime("%H")
    df_games_csv.to_csv('game_predictions_{0}_{1}_{2}.csv'.format(year,smonth,sday))
    
    return {'games_csv': df_games_csv, 
            'display_obj': df_games_display, 
            'rosters' : df_rosters}

def odds_to_payout(array):

    return 1+ np.where(array > 0, array/100., 100/-array)


def zap_player_pos(dft, player_name, pos_w):

    """helper function to reallocate a players mintues through the rest of the team
    pos because it takes into account the player position when allocating minutes"""

    idx = dft[dft.player_name == player_name].index[0]
    seconds_to_allocate = dft.loc[idx,'seconds_played']*0.75
    drt = np.clip(dft.loc[idx,'raptor_defense']*0.5, 0, 5) * seconds_to_allocate
    position = dft.loc[idx,'position']

    dff = dft[dft.player_name != player_name].copy()
    #= dff[dff.poistion == position].rank.idmax()
    T = 60.

    #higher weight for more similar positions
    pos_score = (1600 - np.abs(400*(dff.position - position)))/60.
    score = (1300 - np.abs(dff.seconds_played - 1200))/T

    w_minutes = np.exp(score)/np.exp(score).sum()
    w_depth = np.exp(pos_score)/np.exp(pos_score).sum()

    #multiply the position and minutes ranking and then re-normalize
    weight = (pos_w * w_depth) * ((1-pos_w) * w_minutes)
    weight = weight/weight.sum()
    
    #print(weight)
    weight.fillna(0,inplace=True)

    dff['seconds_played'] = np.clip(dff['seconds_played'] + (seconds_to_allocate * weight), 0, 60*38.0)
    dff['w_minutes'] = w_minutes
    dff['w_depth'] = w_depth

    dff['ratings_product_offense'] = dff['raptor_offense']*dff['seconds_played']
    dff['ratings_product_defense'] = dff['raptor_defense']*dff['seconds_played']
    dff['ratings_product_defense'] = dff['ratings_product_defense'] - (drt * weight)

    dff['ratings_product_pace'] = dff['pace_impact']*dff['seconds_played']

    return dff


def get_odds():

    date = datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
    cmd = f"wget -O odds.txt \"https://www.rotowire.com/basketball/tables/odds.php?date={date}&odds=stats\""
    os.system(cmd)
    with open ("odds.txt", "r") as myfile:
        data=myfile.read().replace('\n', '')
    data2 = data[1:-1]
    mydict = ast.literal_eval(data2)

    try:
        df = pd.DataFrame.from_dict(mydict)
    except(ValueError):
        df = pd.DataFrame(mydict,index=[0])
    df['favourite'] = df['favourite'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df['underdog'] = df['underdog'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df = df.rename(columns={"favourite": "favorite",
                        "fml": "favorite moneyline",
                        "uml": "underdog moneyline",
                        "ou": "over-under"})


    df = df[['time', 'favorite', 'underdog', 'favorite moneyline', 'underdog moneyline', 'line', 'over-under']]
    abb_mappings  = {'CHA':'CHO','CHR':'CHO', 'NY': 'NYK', 'NO':
       'NOP', 'NOR':'NOP','BRO':'BRK','BKN':'BRK', 'GS': 'GSW','SAN':'SAS'}

    df.favorite = df.favorite.replace(abb_mappings)
    df.underdog = df.underdog.replace(abb_mappings)
    df['favorite moneyline'] = df['favorite moneyline'].replace({'':-9999})
    df['underdog moneyline'] = df['underdog moneyline'].replace({'':-9999})
    df['over-under'] = df['over-under'].replace({'': -9999})


    return df

def add_kelly(games_csv, df_odds):

    #print (df.head())
    f = dict(zip(df_odds.favorite.values, df_odds['favorite moneyline'].values))
    u = dict(zip(df_odds.underdog.values, df_odds['underdog moneyline'].values))
    ut = dict(zip(df_odds.underdog.values, df_odds['over-under'].values))
    ft = dict(zip(df_odds.favorite.values, df_odds['over-under'].values))
    us = dict(zip(df_odds.underdog.values, df_odds['line'].values.astype(float)*-1))
    fs = dict(zip(df_odds.favorite.values, df_odds['line'].values))

    ml = {**f, **u}
    totals_mapping = {**ut, **ft}
    spread_mapping = {**us, **fs}
    abb_map = get_abbreviation_mapping()
    games_csv['team2'] = games_csv['AWAY_team'].map(abb_map)
    games_csv['team1'] = games_csv['HOME_team'].map(abb_map)
    games_csv['line1'] = games_csv['team1'].map(ml)
    games_csv['line2'] = games_csv['team2'].map(ml)
    games_csv['over_under'] = games_csv['team2'].map(totals_mapping).astype(float)
    games_csv['spread1'] = games_csv['team1'].map(spread_mapping).astype(float)
    games_csv['spread2'] = games_csv['team2'].map(spread_mapping).astype(float)
    
    games_csv['HOME_wins_p'] = 1-games_csv['away_win_p']
    games_csv['spread_p'] = games_csv['model_away_p'] - games_csv['model_home_p']

    dff = games_csv[['team1','team2','line1','line2','spread1','spread2',
                     'model_home_p','model_away_p','spread_p','HOME_wins_p','over_under']].copy()
    dff['team2_win_prob'] = 1.0 - dff['HOME_wins_p']

    g1 = dff[['team1','team2','line1','spread1','spread_p','HOME_wins_p','over_under']].copy()

    g1['predicted_score'] = np.round(dff['model_home_p'],1).astype(str) +'-' + np.round(dff['model_away_p'],1).astype(str)
    g1['predicted_total'] = np.round(dff['model_away_p']+dff['model_home_p'],1)

    g1.columns = ['team','opponent','line','spread','predicted_spread','win_prob','over_under','predicted_score','predicted_total']

    g2 = dff[['team2','team1','line2','spread2','spread_p','team2_win_prob','over_under']].copy()

    g2['spread_p'] = -1*g2['spread_p']
    g2['predicted_score'] = np.round(dff['model_away_p'],1).astype(str) +'-' + np.round(dff['model_home_p'],1).astype(str)
    g2['predicted_total'] = np.round(dff['model_away_p']+dff['model_home_p'],1)

    g2.columns = ['team','opponent','line','spread','predicted_spread','win_prob','over_under','predicted_score','predicted_total']
    g1['location'] = 'home'
    g2['location'] = 'away'

    # split the games into separate betting opps

    df_g = pd.concat([g1,g2])

    #print(df_g.head())

    df_g.line = df_g.line.astype(float)
    df_g.reset_index(inplace=True)

    df_g['odds'] = df_g['line'].apply(odds_conversion)
    df_g['exp_profit']= revenue_long(np.repeat(1,df_g.shape[0]),df_g['odds'],df_g['win_prob'])
    df_g['kelly'] = kelly(df_g['odds'],df_g['win_prob'])

    # apply the strategies
    kellys=pd.DataFrame(kelly_strat(df_g['kelly']))
    softmaxx = pd.DataFrame(softmax_strat_pos_ev_only(df_g['exp_profit'],0.5))

    kellys.columns=['kelly_weights']
    softmaxx.columns=['softmax_weights']

    final=pd.concat([kellys,softmaxx, df_g], axis = 1)

    budget = 10
    final['kelly_weights'] = final['kelly_weights']*budget
    final['softmax_weights'] = final['softmax_weights']*budget

    final.sort_values('kelly_weights', ascending = False, inplace=True)
    #for now, bet if more than 2 points different
    final['O/U bet'] = np.where((final['predicted_total'] > final['over_under']+4.9),'Over',
                                       np.where(final['predicted_total'] < final['over_under']-4,
                                       'Under', 'None'))

    final['spread_margin'] = final['spread']-final['predicted_spread']

    spread_bins = [-999, 1, 3, 5, 9999]
    spread_labels = ['None', 'Small', 'Medium', 'Large']

    final['spread bet'] = pd.cut(final['spread_margin'], bins=spread_bins, labels=spread_labels).astype(str)

    for col in ['kelly_weights','win_prob','odds','exp_profit']:

        final[col] = np.round(final[col],3)

    return final[['team','opponent','kelly_weights','location','exp_profit','win_prob','odds','line','spread','predicted_score','over_under','O/U bet','spread bet']]

def eval_portfolio(w,df):
    outcome = df.win_prob.apply(simulate_game)
    revenue = revenue_long(w,df['odds'],outcome)
    return revenue

def softmax_strat_pos_ev_only(ev,temp):
    ev = pd.Series(ev).copy()
    ev.loc[ev<0] = np.nan
    w = np.exp(ev/temp)
    w = w/np.sum(w)
    w.loc[w.isna()] = 0
    return w

def softmax_strat_all(ev,temp):
    ev = pd.Series(ev).copy()
    w = np.exp(ev/temp)
    w = w/np.sum(w)
    w.loc[w.isna()] = 0
    return w

def kelly_strat(ev):
    ev = pd.Series(ev).copy()
    ev.loc[ev<0] = np.nan
    w = ev.copy()
    w.loc[w.isna()] = 0
    return w

def eval_real_portfolio(w,df):
    outcome = df.winner.values
    revenue = revenue_long(w,df['odds'],outcome)
    return revenue

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

def get_game_stats(team, game_date):
    
    game_cols = ['game_index','away_points','home_points','away_offensive_rating',
     'home_offensive_rating','pace','date']
    hmtl_str = 'https://www.basketball-reference.com/teams/{}/2021/gamelog-advanced/'.format(team)
    # modified for nba playoffs, data comes in as a separate table
    # hmtl_str = 'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fteams%2F{}%2F2020%2Fgamelog-advanced%2F&div=div_tgl_advanced_playoffs'.format(team)
    team_game_log = pd.read_html(hmtl_str, skiprows=1, header=0)
    df_log = team_game_log[0]
    # print(team, game_date, hmtl_str)
    game_stats = df_log[df_log.Date == game_date].iloc[0]
    home_otr = float(game_stats.ORtg)
    away_otr = float(game_stats.DRtg)
    home_points = game_stats.Tm
    away_points = game_stats['Opp.1']
    pace = float(game_stats.Pace)
    date = game_stats.Date
    game_index = date.replace('-','') + '0' + team
    row = pd.DataFrame(data=np.array([game_index,away_points,
                                  home_points,away_otr,home_otr,
                                  pace, date]).reshape(1,-1), columns=game_cols)
    #row['date'] = pd.to_datetime(row['date'])
    return row

def construct_sup_features(season):
    
    """function is used to create efficiency and pace season averages
    and also get adjustment for OT games """
    
    dfs = pd.read_csv('../data/{}_season_data.csv'.format(season))
    #dfs = pd.read_csv('data/{}_season_data.csv'.format(season))
    dfs.rename(columns=({'Unnamed: 0':'game_index'}),inplace=True)
    dfs['poss'] = 100*dfs['away_points']/dfs['away_offensive_rating']
    dfs['est_minutes'] = dfs['poss']*48/dfs['pace']
    dfs['score_norm'] = np.round(48/dfs['est_minutes'],2)
    dfs = dfs[dfs.team == dfs['game_index'].map(lambda x: x[-3:])]
    year = dfs['game_index'].map(lambda x: x[:4])
    month = dfs['game_index'].map(lambda x: x[4:6])
    day = dfs['game_index'].map(lambda x: x[6:8])
    dfs['date'] = pd.to_datetime(year + '-' + month + '-' + day)
    # removes playoffs and last game of season
    dfs = dfs[dfs.date < '{}-04-10'.format(season)]
    means = dfs[['pace','home_offensive_rating','away_offensive_rating']].mean()
    dfs['pace_mean'] = means['pace']
    dfs['home_rate_mean'] = means['home_offensive_rating']
    dfs['away_rate_mean'] = means['away_offensive_rating']
    dfs[['game_index','score_norm','pace_mean','home_rate_mean',
         'away_rate_mean']].to_csv('data/{}_reg_sup.csv'.format(season),index=False)

def run_prediction_script():
    
    #define constants
    points_features = ['AWAY_p_allowed', 'HOME_p_allowed', 'AWAY_p_scored',
                       'HOME_p_scored', 'combined_pace','abs_eff_diff']

    prob_features = ['diff'] + ['model_total_p']

    #previous seasons data
    dfr, dfp = get_historical_data([2016, 2017, 2018, 2019])
    dfp['act_seconds_played'] = dfp['seconds_played']
    dfrr = pd.DataFrame()
    for season in [2016, 2017, 2018, 2019]:
        dfr_temp = dfr[dfr.season == season].copy()
        dfr_temp = regularize_raptor(dfr_temp, season, 100)
        dfrr = dfr_temp.append(dfrr)
    
    dfp = project_seconds_played(dfp)

    #I use this to get efficiencies for each season
    df_sup = get_sup_data([2016, 2017, 2018, 2019])
    #print(dfp.game_index.nunique())
    #print(df_sup.game_index.nunique())

    #print(type(dfp.game_index.iloc[0]))
    #print(type(df_sup.game_index.iloc[0]))

    dfr_games = raptor_features_season_v2(dfp, dfrr, df_sup, .78)

    #current seasons dta
    dfr_current, dfp_current, dates = get_current_data()

    dfr_current_adj = regularize_raptor(dfr_current, 2021, 140)
    dfp_current = project_seconds_played(dfp_current)
    dfp_adj = get_inj_adjusted_minutes(dfp_current)
    dfe = get_latest_effiency_data(dates)
    df_sup_current = make_sup_current(dfe)
    dfr_games_curr = raptor_features_season_v2(dfp_adj, dfr_current_adj, df_sup_current, .78)

    #put all the data together
    df_today, df_roster = make_today_features(dfp_adj, dfr_current_adj, df_sup_current, days=6)
    dfg = dfr_games.append(dfr_games_curr)
    dfg = dfr_games.append(df_today)
    dfg = add_dummies(dfg)

    #split the data
    today = pd.Timestamp(datetime.now(pytz.timezone('US/Pacific')).date())
    df_train = dfg[dfg.date < today].copy()
    df_today = dfg.drop(df_train.index).copy()

    #train data
    df_train, points_model, prob_model = train_models(df_train, points_features, prob_features)

    #get points and probabilities
    df_today = pred_points(df_today, points_model, points_features)
    df_today = pred_probs(df_today, prob_model, prob_features)

    #add odds for bets
    df_odds = get_odds()
    df_kelly = add_kelly(df_today, df_odds)
    
    results = {'kelly': df_kelly, 'predictions': df_today, 'roster': df_roster}

    return results