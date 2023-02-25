##########################################################################################################
########             LIBRAIRIES                                                                   ########
##########################################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf  # https://pypi.org/project/yfinance/
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

###########################################################################################################
#########             FONCTIONS : PARTIE I                                                         ########
###########################################################################################################

def EMA(df, n):
    EMA = pd.Series(df['Adj Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA

def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC

def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period]) 
    d = d.drop(d.index[:(period - 1)])
    rs = u.ewm(com=period - 1, adjust=False).mean() / \
         d.ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD

def getlastdata(ticker, j, optionsEMASTORSI, optionsROCMOM):
    # data = yf.download(ticker, start=datetime.today() - timedelta(days=60),end=datetime.now(),interval=intervaltemps)["Adj Close"]
    data = yf.download(ticker, start=datetime.today() - timedelta(days=60), end=datetime.now(), interval="2m")
    data = pd.DataFrame(data)
    index = data.index + pd.DateOffset(hours=1)
    data = data.reset_index(drop=True)
    data["Datetime"] = index
    data = data.set_index('Datetime')

    data["returns"] = np.log(data["Adj Close"].div(data["Adj Close"].shift(1)))
    data.dropna(inplace=True)
    data["direction"] = np.sign(data.returns)
    data = data[data.direction != 0.0]
    lags = j
    cols = []

    for lag in range(1, lags + 1):
        col = "lag{}".format(lag)
        data[col] = data.returns.shift(lag)
        cols.append(col)
    data.dropna(inplace=True)
    
    data['short_mavg'] = data['Adj Close'].rolling(window=10, min_periods=1, center=False).mean()
    data['long_mavg'] = data['Adj Close'].rolling(window=60, min_periods=1, center=False).mean()
    data['signal'] = np.where(data['short_mavg'] > data['long_mavg'], 1.0, 0.0)

    for i in range(len(optionsEMASTORSI)):
        data["EMA{}".format(optionsEMASTORSI[i])] = EMA(data, optionsEMASTORSI[i])
        data["RSI{}".format(optionsEMASTORSI[i])] = RSI(data['Adj Close'], optionsEMASTORSI[i])
        data["%K{}".format(optionsEMASTORSI[i])] = STOK(data['Adj Close'], data['Low'], data['High'], optionsEMASTORSI[i])
        data["%D{}".format(optionsEMASTORSI[i])] = STOD(data['Adj Close'], data['Low'], data['High'], optionsEMASTORSI[i])
    for i in range(len(optionsROCMOM)):
        data["ROC{}".format(optionsROCMOM[i])] = ROC(data['Adj Close'], optionsROCMOM[i])
        data["MOM{}".format(optionsROCMOM[i])] = MOM(data['Adj Close'], optionsROCMOM[i])

    del data["Close"]
    del data["Open"]
    del data["High"]
    del data["Low"]

    for x in ['Volume']:  # la variable volume contient globalement le plus d'outliers d'où le filtre en particuliers sur cette variable
        q75, q25 = np.percentile(data.loc[:, x], [75, 25])
        intr_qr = q75 - q25

        max = q75 + (1.5 * intr_qr)
        min = q25 - (1.5 * intr_qr)

        data.loc[data[x] < min, x] = np.nan
        data.loc[data[x] > max, x] = np.nan

    data = data.dropna(axis=0)
    
    return data, cols

def MLfit(data, size, num_folds):

    X = data.loc[:, data.columns != 'direction']
    y = data.direction
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size)

    models = []

    models.append(('Logistic Regression', LogisticRegression()))
    models.append(('Linear Discrimant Analysis', LinearDiscriminantAnalysis()))
    models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
   # models.append(('Support Vector Classification', SVC()))
    # #Neural Network
   # models.append(('Multi-layer perceptron', MLPClassifier()))
    # # Boosting methods
    models.append(('Gradient Boosting Classifier', GradientBoostingClassifier()))
    # # Bagging methods
    models.append(('Random Forest Classifier', RandomForestClassifier()))


    scoring = 'accuracy'
    resultsTrain = []
    resultsTest = []
    names = []

    for name, model in models:
        names.append(name)
        kfold = KFold(n_splits=num_folds)
        cv_resultsTrain = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        resultsTrain.append(cv_resultsTrain)
        cv_resultsTest = cross_val_score(model, X_test, Y_test, cv=kfold, scoring=scoring)
        resultsTest.append(cv_resultsTest)

        msg = "{}: {} ({})".format(name, round(cv_resultsTrain.mean(), 3), round(cv_resultsTrain.std() ,3))
        st.write(msg)

    return resultsTrain, resultsTest, names, X_train, Y_train, num_folds, scoring, X_test, Y_test

def GridSearchForRF(X_train,Y_train, X_test, Y_test, num_folds):
    
    scoring = 'accuracy'
    n_estimators = [20, 80]
    max_depth = [5, 10]
    criterion = ["gini", "entropy"]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    model = RandomForestClassifier(n_jobs=-1)
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)

    best_paramRF = grid_result.best_params_

    modelRF = RandomForestClassifier(criterion=best_paramRF["criterion"], n_estimators=best_paramRF["n_estimators"],
                                   max_depth=best_paramRF["max_depth"],
                                   n_jobs=-1)
    modelRF.fit(X_train, Y_train)
    predictions = modelRF.predict(X_test)
    df_cm = pd.DataFrame(confusion_matrix(Y_test, predictions), columns=np.unique(Y_test),
                         index=np.unique(Y_test))

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig5 = plt.figure(figsize=(15, 7))
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font sizes
    st.write("##### Matrice de confusion RF :")
    st.pyplot(fig5)

    return best_paramRF["criterion"], best_paramRF["n_estimators"], best_paramRF["max_depth"], modelRF

def GridSearchForLR(X_train,Y_train, X_test, Y_test):
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    logistic = LogisticRegression()
    grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=5, verbose=2)
    grid_result = grid_search.fit(X_train, Y_train)
    best_C = grid_result.best_params_['C']
    best_penalty = grid_result.best_params_['penalty']

    modellr = LogisticRegression(C=best_C, penalty=best_penalty)
    modellr.fit(X_train, Y_train)
    predictionslr = modellr.predict(X_test)
    df_cm = pd.DataFrame(confusion_matrix(Y_test, predictionslr), columns=np.unique(Y_test),
                         index=np.unique(Y_test))

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig6 = plt.figure(figsize=(15, 7))
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font sizes
    st.write("##### Matrice de confusion LR :")
    st.pyplot(fig6)

    return best_C, best_penalty, modellr

def GridSearchForGBM(X_train,Y_train, num_folds, X_test, Y_test):
    
    scoring = 'accuracy'
    n_estimators = [20, 180, 1000]
    max_depth = [2, 3, 5]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    model = GradientBoostingClassifier()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    best_paramGBM = grid_result.best_params_

    modelGBM =GradientBoostingClassifier(max_depth=best_paramGBM["max_depth"],n_estimators=best_paramGBM["n_estimators"])
    modelGBM.fit(X_train, Y_train)
    #predictions = modelGBM.predict(X_test)
    #df_cm = pd.DataFrame(confusion_matrix(Y_test, predictions), columns=np.unique(Y_test),
      #                   index=np.unique(Y_test))

     #df_cm.index.name = 'Actual'
     #df_cm.columns.name = 'Predicted'
     #fig5 = plt.figure(figsize=(15, 7))
    # sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font sizes
    # st.write("##### Matrice de confusion GBM :")
    # st.pyplot(fig5)

    return best_paramGBM["max_depth"], best_paramGBM["n_estimators"], modelGBM

###########################################################################################################
#########             FONCTIONS : PARTIE II                                                         #######
###########################################################################################################

def mlAlgoPred(data, forecast_out,model, size, criterion, n_estimators,max_depth, best_paramGBM_max_depth, best_paramGBM_n_estimators, best_C, best_penalty, type=False):

    X = np.array(data.loc[:, data.columns != 'direction'])
    X = X[:-forecast_out]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = np.array(data[["direction"]].shift(-forecast_out))
    y = y[:-forecast_out]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=size)
    x_forecast = np.array(X)[-forecast_out:]

    ####### AJOUTER LES ALGORITHMES ICI #########################################
    if type == False:

        rf = RandomForestClassifier(criterion =criterion, n_estimators =n_estimators,max_depth=max_depth,  n_jobs=-1)
        modelGBM = GradientBoostingClassifier(max_depth=best_paramGBM_max_depth,n_estimators=best_paramGBM_n_estimators)
        modellr = LogisticRegression(C=best_C, penalty=best_penalty)

        ##############################################################################

        liste = [rf, modelGBM, modellr]  # modifier les listes si ajout d'un algorithme
        liste2 = ["RF","GBM" ,"LR"]  # modifier les listes si ajout d'un algorithme
        conf = []
        prediction = []

        #################################################################################

        for i in liste:
            i.fit(x_train, y_train)
            conf.append(i.score(x_test, y_test))
            prediction.append(i.predict(x_forecast))

        for n in range(len(liste)):
            print("Score de confiance de", list(liste)[n], "est ", conf[n])

        test = pd.DataFrame([])

        prediction1 = pd.DataFrame(prediction).T

        for i in range(0, len(prediction1.columns)):
            test[i] = pd.DataFrame(prediction1[i])

        prediction1.columns = [col + '_prediction ' for col in liste2]

    if type==True: # ce paramètre est utilisé si on charge un fichier pickle 

        ##############################################################################
        liste = [model]
        liste2 = ["Modèle"]
        #################################################################################
        prediction = []

        for i in liste:
            prediction.append(i.predict(x_forecast))

        test = pd.DataFrame([])

        prediction1 = pd.DataFrame(prediction).T

        for i in range(0, len(prediction1.columns)):
            test[i] = pd.DataFrame(prediction1[i])

        prediction1.columns = [col + '_prediction ' for col in liste2]

    return prediction1


def OperationOnDF(data, prediction1, forecast_out, type=False):
    
    somme = pd.DataFrame(data["direction"])
    for i in list(prediction1):
        somme[i] = somme['direction'].append(prediction1[i])

    datelist = pd.date_range(datetime.now(), periods=forecast_out, freq="2min").tolist()
    df = pd.DataFrame(datelist)
    newdata = somme.reset_index()

    date = newdata["Datetime"]
    somme = somme.reset_index()
    del somme["Datetime"]
    del somme["direction"]
    date = pd.DataFrame(date)
    somme1 = somme.append(prediction1)
    somme1 = somme1.reset_index()
    del somme1["index"]

    date = pd.DataFrame(date)
    date = date.Datetime.dt.strftime("%Y-%m-%d %H:%M:%S")

    df.columns = ['Date']
    df = df.Date.dt.strftime("%Y-%m-%d %H:%M:%S")

    date = pd.DataFrame(date)
    test = date.append(df)
    test = test.reset_index()
    del test['index']
    test = test.Datetime.dropna()
    date = test.append(df)
    date = pd.DataFrame(date)
    date.columns = ['Date']
    date = date.reset_index()
    del date['index']
    prediction = pd.concat([date, somme1], axis=1)
    prediction = prediction.set_index(["Date"])

    if type == False:
        
        conscensus = prediction.tail(forecast_out)
        conscensus = pd.DataFrame(conscensus)
        conscensus['RF_direction'] = pd.np.where(conscensus['RF_prediction '] == 1, "buy",
                                                 pd.np.where(conscensus['RF_prediction '] == 0, "neutral", "sell"))
        conscensus['GBM_direction'] = pd.np.where(conscensus['GBM_prediction '] == 1, "buy",
                                                  pd.np.where(conscensus['GBM_prediction '] == 0, "neutral", "sell"))
        conscensus['LR_direction'] = pd.np.where(conscensus['LR_prediction '] == 1, "buy",
                                                 pd.np.where(conscensus['LR_prediction '] == 0, "neutral", "sell"))
        conscensus = conscensus.reset_index()
        conscensus = conscensus.set_index('Date')
        conscensus.index = pd.to_datetime(conscensus.index)
        conscensus.index = conscensus.index+ pd.DateOffset(hours=1)
        conscensus = conscensus.reset_index()
        
        
        
        conscensus2 = prediction.median(axis=1)
        conscensus2 = pd.DataFrame(conscensus2)
        conscensus2.columns = ['conscensus']
        conscensus2 = conscensus2.dropna()
        conscensus2 = pd.DataFrame(conscensus2["conscensus"].astype(int))
        conscensus2['direction'] = pd.np.where(conscensus2['conscensus'] == 1, "buy",pd.np.where(conscensus2['conscensus'] == 0, "neutral", "sell"))
        conscensus2 = conscensus2.reset_index()
        conscensus2 = conscensus2.tail(forecast_out)
        conscensus2 = conscensus2.set_index('Date')
        conscensus2.index = pd.to_datetime(conscensus2.index)
        conscensus2.index = conscensus2.index+ pd.DateOffset(hours=1)
        conscensus2 = conscensus2.reset_index()

    if type == True: # ce paramètre est utilisé si on charge un fichier pickle 

        conscensus = prediction.tail(forecast_out)
        conscensus = pd.DataFrame(conscensus)
        conscensus['direction'] = pd.np.where(conscensus['Modèle_prediction '] == 1, "buy", pd.np.where(conscensus['Modèle_prediction '] == 0, "neutral", "sell"))
        conscensus = conscensus.reset_index()
        conscensus2 = []

    return somme1, conscensus, conscensus2
