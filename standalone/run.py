from raptor_script_utils import *
from get_wins import *
import pytz

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import streamlit as st

def main():
    
    st.set_page_config(layout='wide', initial_sidebar_state="collapsed")
    
    date = datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
    csvfile = f"kelly_{date}.csv"
    
    st.title(f"Drake's NBA Predictions: {date}")
    
    yest_results, num_games, num_correct, profit, num_today = get_wins()

    
    st.write("Yesterday's results:")
    if (num_games == 0):
        st.write("No games!")
    elif (num_correct == -1):
        st.write("No data available to calculate performance for yesterday!")
    else:
        yest_results['is_winner'] = yest_results['is_winner'].astype(bool)
        yest_results['pred_winner'] = yest_results['pred_winner'].astype(bool)
        st.table(yest_results)
        st.write(f"Yesterday's predictions: {num_correct}/{num_games} predicted corrrectly, profit = {profit}%")

    if (num_today == 0):
        st.write('Rest day for Drake: No games today!')
    else:
        results = run_prediction_script()
        st.write("Today's predictions:")
        results['kelly'].to_csv(csvfile, index=None)
        results['kelly'] = results['kelly'].drop(['exp_profit', 'odds'], axis=1)
        df = results['kelly'].copy()
        st.table(df)
        
        injured = get_injured()
        df3 = injured[injured['abb'].isin(df.team.values)]
        df3 = df3[['abb', 'name']].sort_values(by=['abb']).copy()
        st.write("Injuries known to Drake (for today's games):")
        st.table(df3)
        
        print(f"Saved results to {csvfile}")
    

if __name__ == "__main__":
    main()
