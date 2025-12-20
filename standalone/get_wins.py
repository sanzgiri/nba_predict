from raptor_script_utils import *
from daily_weights_selector import *
from basketball_reference_web_scraper import client
import pytz
import os

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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
         "PHOENIX SUNS": "PHO",
         "PORTLAND TRAIL BLAZERS": "POR",
         "SACRAMENTO KINGS": "SAC",
         "SAN ANTONIO SPURS": "SAS",
         "TORONTO RAPTORS": "TOR",
         "UTAH JAZZ": "UTA",
         "WASHINGTON WIZARDS": "WAS"}

def get_wins():

    today = datetime.now(pytz.timezone('US/Pacific'))
    yesterday = (datetime.now(pytz.timezone('US/Pacific'))-timedelta(days=1))
    
    box_scores = client.team_box_scores(day=yesterday.day, month=yesterday.month, year=yesterday.year)
    num_teams = len(box_scores)
    num_games = int(num_teams/2)
    
    season_obj = client.season_schedule(season_end_year=2021)
    todays_games = [g for g in season_obj if utc_to_local(g['start_time']).date()
                    == datetime(today.year, today.month, today.day).date()]
    num_today = len(todays_games)
    
    csvfile = f'kelly_{yesterday.strftime("%Y-%m-%d")}.csv'
    csvexists = os.path.exists(csvfile)
    
    if num_games == 0:
        return pd.DataFrame(), num_games, 0, 0, num_today
        
    if csvexists == False:
        return pd.DataFrame(), num_games, -1, 0, num_today
    
    resfile = f'results_{yesterday.strftime("%Y-%m-%d")}.csv'
    dfk = pd.read_csv(csvfile)
    
    if 'odds'not in dfk.columns:
        return pd.DataFrame(), num_games, -1, 0, num_today
    
    bs_dict={}
    for i in range(num_teams):
        bs = box_scores[i]
        team = abbr2[bs['team'].value]
        score = (bs['made_field_goals'] - bs['made_three_point_field_goals'])*2 + bs['made_three_point_field_goals']*3 + bs['made_free_throws']
        bs_dict[team] = score
        
    for i in range(len(dfk)):
        dfk.loc[i, 'pred_winner'] = dfk['win_prob'].iloc[i] >= 0.5
        try:
            dfk.loc[i, 'is_winner'] = bs_dict[dfk.team.iloc[i]] > bs_dict[dfk.opponent.iloc[i]]
        except(KeyError):
            dfk.loc[i, 'is_winner'] = np.nan
    
    dfk = dfk[dfk.is_winner.notnull()].copy()
    dfk['is_winner'] = dfk['is_winner'].astype(bool)
    num_correct = len(dfk[(dfk.is_winner == True) & (dfk.win_prob >= 0.5)])
    df = dfk[['team', 'opponent','location','win_prob', 'pred_winner','is_winner', 'odds', 'kelly_weights']].copy()
    #df = df[df.is_winner==True].copy()
    
    dfk = dfk[dfk['kelly_weights'] > 0].copy()
    dfk['multiplier'] = dfk['is_winner'].apply(lambda x: 1 if x else 0)
    dfk['earnings'] = dfk['kelly_weights']*dfk['odds']*dfk['multiplier']
    dfk = dfk[['team', 'opponent', 'kelly_weights', 'odds', 'is_winner', 'earnings']].copy()
    dfk.to_csv(resfile, index=None)

    profit_pct = int(100*(dfk.earnings.sum()/dfk.kelly_weights.sum()-1))
    
    
    
    return df, num_games, num_correct, profit_pct, num_today
    


