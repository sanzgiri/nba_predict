import pandas as pd
from datetime import date, datetime, timezone, timedelta

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

def utc_to_local(utc_dt):
    
    """utility to get datetime of games in a different time zone"""
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)

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
       'NOP', 'BKN':'BRK', 'GS': 'GSW'}

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

    seconds_projection = (df_temp.groupby('slug')[['seconds_played','seconds_played_adj']].mean()).reset_index()
    
    dft = pd.merge(seconds_projection, dfr, left_on = 'slug', right_on = 'player_id')
    
    dft['ratings_product_offense'] = dft['raptor_offense']*dft['seconds_played']
    dft['ratings_product_defense'] = dft['raptor_defense']*dft['seconds_played']
    dft['ratings_product_pace'] = dft['pace_impact']*dft['seconds_played']
    
    dft['team'] = team
    
    df_dc = pd.read_csv('../data/team_depth_charts.csv').iloc[:,1:]
    dftm = pd.merge(df_dc, dft, left_on = 'name', right_on = 'player_name', how='right')

    #if missing position, make them SF
    dftm.position.fillna(2.0, inplace=True)

    players_to_zap = dftm[dftm.seconds_played_adj == 0].player_name.unique()
    if len(players_to_zap) > 0:
        for player in players_to_zap:
            dftm = zap_player_pos(dftm, player, .75)
        
    return dftm

def game_agg(dft_home, dft_away, adj, avg_ort, avg_pace):

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
    
    dft_agg = df_teams.groupby(['location','team'])[cols].sum()
    dft_agg['game_index'] =  home_team + '_v_' + away_team
    
    dfg = game_raptor_calcs(dft_agg, adj, avg_ort, avg_pace)
    
    return dfg

def game_raptor_calcs(dft, adj, avg_ort, avg_pace):
    
    """compute raptor featuers needed for model"""
    
    PLAYERS = 5
    ADJ = adj
    AVG_ORT = avg_ort
    AVG_PACE = avg_pace
    
    dft['ORT'] = ADJ * PLAYERS * (dft['ratings_product_offense']/dft['seconds_played'])
    dft['DRT'] = ADJ * PLAYERS * (dft['ratings_product_defense']/dft['seconds_played'])
    dft['pace'] = ADJ*PLAYERS*(dft['ratings_product_pace']/dft['seconds_played'])
    dft['p_scored'] = (AVG_ORT + dft['ORT']) * (dft['pace'] + AVG_PACE)/100. 
    dft['p_allowed'] = (AVG_ORT - dft['DRT']) * (dft['pace'] + AVG_PACE)/100.
    
    #reshape to one row per game
    game_data = dft.reset_index().set_index(['game_index','location'])[['p_allowed','p_scored','team']].unstack()
    
    #format the reshaped data
    data = game_data.values
    vals = game_data.columns.levels[0]
    home_away = game_data.columns.levels[1]
    col_names = [loc + '_' + val for val in vals for loc in home_away]

    dfg = pd.DataFrame(data = game_data.values, columns = col_names, index = game_data.index)
    
    return dfg



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
    
    return dff

def regularize_raptor(dfr, prior_weight):
    
    """smoothes raptor ratings to the replacement level mean
    
    args: 
    
    dfr: the raptor weights
    prior_weight: how much to smooth to the mean"""
    dfrp = pd.read_csv('../data/raptors_player_stats.csv')
    dfrp = dfrp[dfrp.season==2019].copy()
    dfrm = pd.merge(dfr, dfrp, how='left', on = 'player_id', suffixes=('', '_p'))
    dfrm.mp_p.fillna(0, inplace=True)
    dfrm.raptor_offense_p.fillna(0, inplace=True)
    dfrm.raptor_defense_p.fillna(0, inplace=True)
    dfrm.pace_impact_p.fillna(0, inplace=True)
    off_pw = (200./(200.+dfrm['mp'].astype(float)))
    def_pw = (50./(50.+dfrm['mp'].astype(float)))

    
    dfrm['raptor_offense'] = dfrm['raptor_offense'] * (1-off_pw) + dfrm['raptor_offense_p'] * (off_pw)
    dfrm['raptor_defense'] = dfrm['raptor_defense'] * (1-def_pw) + dfrm['raptor_defense_p'] * (def_pw)
    dfrm['pace_impact'] = dfrm['pace_impact'] * (1-def_pw) + dfrm['pace_impact_p'] * (def_pw)

    
    lower_quantile = dfrm.mp.quantile(q=.25)
    
    df_scrubs = dfrm[dfrm.mp < lower_quantile ] 
    
    repl_off = (df_scrubs.mp * df_scrubs.raptor_offense).sum()/(df_scrubs.mp.sum())
    repl_def = (df_scrubs.mp * df_scrubs.raptor_defense).sum()/(df_scrubs.mp.sum())
    repl_pace = (df_scrubs.mp * df_scrubs.pace_impact).sum()/(df_scrubs.mp.sum())
    
    sw = (prior_weight/(prior_weight+dfrm['mp'].astype(float) + .7*dfrm['mp_p']))

    dfrm['raptor_offense'] = ((dfrm['raptor_offense'] *(1-sw))  
                                         + (repl_off * sw))
    dfrm['raptor_defense'] = ((dfrm['raptor_defense'] *(1-sw))  
                                         + (repl_def * sw))
    dfrm['pace_impact'] = ((dfrm['pace_impact'] *(1-sw))  
                                      + (repl_pace * sw))
    
    return dfrm


def get_abbreviation_mapping():
    
    """help function to map between different team names"""
    
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
    year = 2020
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


def get_recent_player_minutes():
    
    """pulls data from basketball reference to get recent playing data"""
    
    d = get_abbreviation_mapping()
    dates = get_season_game_dates(2020)
    df_cached = pd.read_csv('../data/recent_player_minutes.csv')
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

        df_cached.append(dfl)
        df_cached.to_csv('../data/recent_player_minutes.csv', index=None)
        
    return df_cached


def get_latest_raptor():
    
    """pull raptor ratings from 538 and save"""
    
    today = datetime.now().date()
    year = today.year
    month = today.month
    day = today.day
    df = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/2020/latest_RAPTOR_by_team.csv')
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
    dfr = dfr[dfr.season.isin(season_list)]
    dfp = pd.DataFrame()
    for season in season_list:
        df_temp = pd.read_csv('../data/{}_player_boxscores.csv'.format(season))
        dfp = dfp.append(df_temp)
    return dfr, dfp

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
                           how = 'left', on = ['name','abb'])
    dfp_ci['seconds_played_adj'] = dfp_ci['seconds_played'] * dfp_ci.player_id.isnull().astype(int)
    
    return dfp_ci
    

def get_current_data():

    dfr_current = get_latest_raptor()
    dfp_current = get_recent_player_minutes()
    #dfp_current.reset_index(inplace=True)
    #dfp_current.rename(columns = {'index':'game_index'},inplace=True)
    
    return dfr_current, dfp_current 

def predict_todays_games(dfp_adj, dfr_current, model, prob_model, features,
                         raptor_slope, avg_ort, avg_pace, days=3):
    
    #get data needed to predict today's games
    #today = datetime.now().date()
    today = date.today() + timedelta(days=1)
    month = today.month
    year = today.year
    day = today.day
    season_year = year if month < 9 else year + 1
    season_obj = client.season_schedule(season_end_year=season_year)

    todays_games = [g for g in season_obj if g['start_time'].date()
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
    seconds_to_allocate = dft.loc[idx,'seconds_played']
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

    dff['seconds_played'] = dff['seconds_played'] + seconds_to_allocate * weight
    dff['w_minutes'] = w_minutes
    dff['w_depth'] = w_depth

    dff['ratings_product_offense'] = dff['raptor_offense']*dff['seconds_played']
    dff['ratings_product_defense'] = dff['raptor_defense']*dff['seconds_played']
    dff['ratings_product_pace'] = dff['pace_impact']*dff['seconds_played']

    return dff

def add_kelly(games_csv):

    #date = datetime.datetime.now().strftime("%Y-%m-%d")
    date = '2019-12-18'
    cmd = f"wget -O odds.txt \"https://www.rotowire.com/basketball/tables/odds.php?date={date}&odds=stats\""
    os.system(cmd)
    with open ("odds.txt", "r") as myfile:
        data=myfile.read().replace('\n', '')
    data2 = data[1:-1]
    mydict = ast.literal_eval(data2)
    df = pd.DataFrame.from_dict(mydict)
    df['favourite'] = df['favourite'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df['underdog'] = df['underdog'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df = df.rename(columns={"favourite": "favorite",
                        "fml": "favorite moneyline",
                        "uml": "underdog moneyline",
                        "ou": "over-under"})
    df = df[['time', 'favorite', 'underdog', 'favorite moneyline', 'underdog moneyline', 'line', 'over-under']]
    abb_mappings  = {'CHA':'CHO','CHR':'CHO', 'NY': 'NYK', 'NO':
       'NOP', 'NOR':'NOP','BRO':'BRK','BKN':'BRK', 'GS': 'GSW'}

    df.favorite = df.favorite.replace(abb_mappings)
    df.underdog = df.underdog.replace(abb_mappings)
    #print (df.head())
    f = dict(zip(df.favorite.values, df['favorite moneyline'].values))
    u = dict(zip(df.underdog.values, df['underdog moneyline'].values))
    ml = {**f, **u}
    abb_map = get_abbreviation_mapping()
    games_csv = results['games_csv']
    games_csv['team2'] = games_csv['AWAY_team'].map(abb_map)
    games_csv['team1'] = games_csv['HOME_team'].map(abb_map)
    games_csv['line1'] = games_csv['team1'].map(ml)
    games_csv['line2'] = games_csv['team2'].map(ml)

    dff = games_csv[['team1','team2','line1','line2','HOME_points_p','AWAY_points_p','HOME_wins_p']].copy()
    dff['team2_win_prob'] = 1.0 - dff['HOME_wins_p']
    g1 = dff[['team1','team2','line1','HOME_wins_p']].copy()
    g1['predicted_score'] = np.round(dff['HOME_points_p'],1).astype(str) +'-' + np.round(dff['AWAY_points_p']).astype(str)
    g1.columns = ['team','opponent','line','win_prob','predicted_score']
    g2 = dff[['team2','team1','line2','team2_win_prob']].copy()
    g2['predicted_score'] = np.round(dff['AWAY_points_p'],1).astype(str) +'-' + np.round(dff['HOME_points_p']).astype(str)

    g2.columns = ['team','opponent','line','win_prob','predicted_score']
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

    for col in ['kelly_weights','win_prob','odds','exp_profit']:

        final[col] = np.round(final[col],3)

    return final[['team','opponent','kelly_weights','exp_profit','win_prob','odds','line','predicted_score']]


def get_odds():

    date = datetime.now().strftime("%Y-%m-%d")
    cmd = f"wget -O odds.txt \"https://www.rotowire.com/basketball/tables/odds.php?date={date}&odds=stats\""
    os.system(cmd)
    with open ("odds.txt", "r") as myfile:
        data=myfile.read().replace('\n', '')
    data2 = data[1:-1]
    mydict = ast.literal_eval(data2)
    df = pd.DataFrame.from_dict(mydict)
    df['favourite'] = df['favourite'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df['underdog'] = df['underdog'].apply(lambda x: re.match(".*team=([A-Z]+).*", x).groups()[0])
    df = df.rename(columns={"favourite": "favorite",
                        "fml": "favorite moneyline",
                        "uml": "underdog moneyline",
                        "ou": "over-under"})
    df = df[['time', 'favorite', 'underdog', 'favorite moneyline', 'underdog moneyline', 'line', 'over-under']]
    abb_mappings  = {'CHA':'CHO','CHR':'CHO', 'NY': 'NYK', 'NO':
       'NOP', 'NOR':'NOP','BRO':'BRK','BKN':'BRK', 'GS': 'GSW'}

    df.favorite = df.favorite.replace(abb_mappings)
    df.underdog = df.underdog.replace(abb_mappings)
    df['favorite moneyline'] = df['favorite moneyline'].replace({'':-9999})
    df['underdog moneyline'] = df['underdog moneyline'].replace({'':-9999})

    return df

def add_kelly(games_csv, df_odds):

    #print (df.head())
    f = dict(zip(df_odds.favorite.values, df_odds['favorite moneyline'].values))
    u = dict(zip(df_odds.underdog.values, df_odds['underdog moneyline'].values))
    ml = {**f, **u}
    abb_map = get_abbreviation_mapping()
    games_csv['team2'] = games_csv['AWAY_team'].map(abb_map)
    games_csv['team1'] = games_csv['HOME_team'].map(abb_map)
    games_csv['line1'] = games_csv['team1'].map(ml)
    games_csv['line2'] = games_csv['team2'].map(ml)

    dff = games_csv[['team1','team2','line1','line2','HOME_points_p','AWAY_points_p','HOME_wins_p']].copy()
    dff['team2_win_prob'] = 1.0 - dff['HOME_wins_p']
    g1 = dff[['team1','team2','line1','HOME_wins_p']].copy()
    g1['predicted_score'] = np.round(dff['HOME_points_p'],1).astype(str) +'-' + np.round(dff['AWAY_points_p']).astype(str)
    g1.columns = ['team','opponent','line','win_prob','predicted_score']
    g2 = dff[['team2','team1','line2','team2_win_prob']].copy()
    g2['predicted_score'] = np.round(dff['AWAY_points_p'],1).astype(str) +'-' + np.round(dff['HOME_points_p']).astype(str)

    g2.columns = ['team','opponent','line','win_prob','predicted_score']
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

    for col in ['kelly_weights','win_prob','odds','exp_profit']:

        final[col] = np.round(final[col],3)

    return final[['team','opponent','kelly_weights','location','exp_profit','win_prob','odds','line','predicted_score']]

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


def run_prediction_script():
    
    """this function should run everything need to run and log"""
    
    features = ['AWAY_p_allowed', 'HOME_p_allowed', 'AWAY_p_scored', 'HOME_p_scored']
    
    #get historical data and run model on historical data
    #yes, some of these constants are a bit magical and seemingly arbitrary
    #at the moment, however, they seem to work well?

    dfr, dfp = get_historical_data([2018])
    rr = linear_model.RidgeCV(alphas=[10.,30.,50.,60.,100.,500.,1000.,3000.], cv=4)
    rr, dfr_games = train_historical(dfp, dfr, .8, 108., 98.91, rr, features)
    
    #get current season and construct features
    dfr_current, dfp_current = get_current_data()
    dfr_current = regularize_raptor(dfr_current, 40.)
    
    df_curr_games = raptor_features_season(dfp_current, dfr_current, .84, 108.9, 99.45)

    #calibrate model for current season
    df_curr_games, prob_model = current_model_calibration(rr, dfr_games, df_curr_games, features)
    
    #get additional data relevant to today's prediction and make predictions
    dfp_adj = get_inj_adjusted_minutes(dfp_current)
    results = predict_todays_games(dfp_adj, dfr_current, rr, prob_model, features, .84, 108.9, 99.45, 4)
    df_odds = get_odds()
    results['kelly'] = add_kelly(results['games_csv'], df_odds)

    return results
    