from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.models import predict
from api.serializers import PredictSerializer

import pandas as pd
import datetime as dt
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
import pickle
import os.path


@api_view(['GET','POST'])
def request_predict(request):
    """
    Request for predict
    """

    with open('store.pkl','rb') as f:  # Python 3: open(..., 'rb')
        pdq = pickle.load(f)
    with open('data.pkl','rb') as f:
        data = pickle.load(f)
    myDict = dict(request.data)
    print(myDict)
    genres = myDict['genres'][0]
    step = int(myDict['steps'][0])
    if step > 500:
        return Response({'result':'error','message':'Too many!'})
    y = data[genres]
    pp,dd,qq,P,D,Q,R = pdq[genres]
    
    mod = sm.tsa.statespace.SARIMAX(y,order=(pp,dd,qq),seasonal_order=(P, D, Q, R),enforce_stationarity=False,enforce_invertibility=False)
    results = mod.fit()

    pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
    pred_ci = pred.conf_int()

    pred_dynamic = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted = pred_dynamic.predicted_mean
    y_truth = y['1998-02-01':]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    # Get forecast x steps ahead in future
    pred_uc = results.get_forecast(steps=step)
    pred_ci = pred_uc.conf_int()
    
    genres_directory = "predict/"+str(genres)
    if not os.path.exists(genres_directory):
        os.makedirs(genres_directory)

    path = "predict/"+str(genres)+"/"+str(step)+".png"
    is_exist = os.path.isfile(path)
    if(not is_exist):
        ax = y.plot(label='observed', figsize=(20, 15))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Counts')
        plt.savefig(path)
        plt.close()

    return Response({'mse':round(mse, 2),'result':'success','path':path})
