from raptor_script_utils import *
from get_wins import *
import pytz

from datetime import datetime, timedelta

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

historic_perf = "../perf/historic_perf.csv"
historic_kelly = "../perf/historic_kelly.csv"

def main():
    
    yesterday = (datetime.now(pytz.timezone('US/Pacific')) - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
    yest_results, num_games, num_correct, profit, num_today = get_wins()

    if (num_games > 0) and (num_correct != -1):
        yest_results.insert(0, 'date', yesterday)
        dfp_in = pd.read_csv(historic_perf)
  
        if yesterday not in dfp_in.date.values:
            print(f"adding perf data for {yesterday}")
            dfp_in = dfp_in.append(yest_results)
            dfp_in.to_csv(historic_perf, index=None)
            
    if (num_today != 0):
        results = run_prediction_script()
        df_results = results['kelly']
        df_results.insert(0, 'date', today)
        
        df_in = pd.read_csv(historic_kelly)
        if today not in df_in.date.values:
            print(f"adding kelly data for {today}")
            df_in = df_in.append(df_results)
            df_in.to_csv(historic_kelly, index=None)

if __name__ == "__main__":
    main()
        