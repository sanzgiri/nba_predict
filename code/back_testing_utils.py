def evaluate_model_rsplit(df, model, N_test, features):
    
    """thiis is for evaluating a model"""
    
    N_games = int(df.shape[0] * .1)
    
    N_folds = int(N_test/N_games)

    
    r = {}
    
    for fold in range(N_folds):
    
        df_train = df.sample(frac=.9)
        df_test = df.drop(df_train.index)

        
        model.fit(df_train[features], df_train[['AWAY_points','HOME_points']])
        
        y_pred_test = model.predict(df_test[features])
        
        mae = np.abs(y_pred_test - df_test[['AWAY_points','HOME_points']]).mean()
        
        accuracy = np.where((y_pred_test[:,0] > y_pred_test[:,1]) == 
                 (df_test['AWAY_points']>df_test['HOME_points']), 1, 0).mean()
        
        r[fold] = {'MAE_away': mae['AWAY_points'],
                        'MAE_home' : mae['HOME_points'],
                        'Accuracy': accuracy}
    
    return pd.DataFrame(r).T.mean()


def evaluate_model_rsplit_per_team(df, model, N_test):
    
    """this didn't seem to perform as well"""
    
    N_games = int(df.shape[0] * .1)
    
    N_folds = int(N_test/N_games)
    
   
    
    r = {}
    
    for fold in range(N_folds):
    
        df_train = df.sample(frac=.9)
        df_test = df.drop(df_train.index)
        
        away_features = ['HOME_p_allowed','AWAY_p_scored']
        
        model.fit(df_train[away_features], df_train['AWAY_points'])
        
        y_pred_away = model.predict(df_test[away_features])
        
        mae_away = np.abs(y_pred_away - df_test['AWAY_points']).mean()
        
        
        home_features = ['HOME_p_scored','AWAY_p_allowed']
        model.fit(df_train[home_features], df_train['HOME_points'])
        
        y_pred_home = model.predict(df_test[home_features])
        
        mae_home = np.abs(y_pred_home - df_test['HOME_points']).mean()
        
        accuracy = np.where((y_pred_away > y_pred_home) == 
                 (df_test['AWAY_points']>df_test['HOME_points']), 1, 0).mean()
        
        r[fold] = {'MAE_away': mae_away,
                        'MAE_home' : mae_home,
                        'Accuracy': accuracy}
    
    return pd.DataFrame(r).T.mean()