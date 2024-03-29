import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import pathlib
import openpyxl
import hypothesis
import pytest
from pandas.conftest import axis
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import os
modelpaths = []
modelpaths.append("predict_3sec_model.h5")
modelpaths.append("predict_6sec_model.h5")
modelpaths.append("predict_10sec_model.h5")
modelpaths.append("predict_30sec_model.h5")
modelpaths.append("predict_60sec_model.h5")
modelpaths.append("predict_120sec_model.h5")
def slidingWindow(ds,window_size,step_size):
    x_data = []
    y_data = []
    for i in range(len(ds)-step_size-window_size):
        x_data.append(ds[i:i+window_size,:])
        #x_data 리스트 = [np.array(window_size,6),np.array(window_size,6),np.array(window_size,6)]
        y_data.append(ds[i+window_size+step_size])
        #y_data 리스트 = [np.array(1),np.array(1),np.array(1),np.array(1)]
    x_data_np = np.reshape(x_data, newshape=(-1, window_size, 6))
    y_data_np = np.reshape(y_data, newshape=(-1, 6))
    return x_data_np, y_data_np
def clip_data(df):
    df['temp1'] = df['temp1'].clip(0, 100) /100
    df['Press1'] = df['Press1'].clip(0, 3.25) /3.25
    df['temp2'] = df['temp2'].clip(0, 100) / 100
    df['Press2'] = df['Press2'].clip(0, 3.25) / 3.25
    df['accel'] = df['accel'].clip(0, 0.1) / 0.1
    df['gas'] = df['gas'].clip(0, 1007) / 1007
    return df
def preprocess_csv(file_path):
    df = pd.read_csv(file_path, header=None, names=['index', 'temp1', 'Press1', 'temp2', 'Press2', 'accel', 'gas'])
    df = clip_data(df)
    return df

new_data = preprocess_csv('_Merged_Data_data_set_01500.csv')

X_new = new_data[['temp1', 'Press1', 'temp2', 'Press2', 'accel','gas']].values
step_sizes =[30, 60, 100, 300, 600, 1200]
window_size = 10
X_news = []
y_news = []
for i in range(len(modelpaths)):
    X, Y = slidingWindow(X_new, window_size, step_sizes[i])
    X_news.append(X)
    y_news.append(Y)

preds = []
for i in range(len(modelpaths)):
    preds.append(tf.keras.models.load_model(modelpaths[i]).predict(X_news[i]))
def showplot(y_new,predictions,step_size):
    # 원본 데이터 그래프
    plt.figure(figsize=(16, 6))
    plt.suptitle(f"{step_size/10}sec_model")
    plt.subplot(1,2,1)
    plt.title("origin")
    for i in range(6):
        plt.plot(y_new[:, i], label=new_data.columns[i+1])
    plt.xlabel('Time(ms)')
    plt.ylabel('Rate[0~1]')
    plt.legend()
    # 예측 데이터 그래프
    plt.subplot(1,2,2)
    plt.title("predict")
    for i in range(6):
        plt.plot(predictions[:, i], label=new_data.columns[i+1])
    plt.xlabel('Time(ms)')
    plt.ylabel('Rate[0~1]')
    plt.legend()
    plt.savefig(f"{int(step_size/10)}sec_origin_predict.png")
for i in range(len(preds)):
    showplot(y_news[i],preds[i],step_sizes[i])


def make_df(y_new,predictions,step_size):
    error = abs(y_new-predictions)
    data = {
        "온도1":[np.mean(error[:,0]),np.min(error[:,0]),np.max(error[:,0])],
        "압력1":[np.mean(error[:,1]),np.min(error[:,1]),np.max(error[:,1])],
        "온도2":[np.mean(error[:,2]),np.min(error[:,2]),np.max(error[:,2])],
        "압력2":[np.mean(error[:,3]),np.min(error[:,3]),np.max(error[:,3])],
        "진동":[np.mean(error[:,4]),np.min(error[:,4]),np.max(error[:,4])],
        "가스":[np.mean(error[:,5]),np.min(error[:,5]),np.max(error[:,5])]
    }
    df = pd.DataFrame(data,index=["평균 Error","Min Error","Max Error"])
    df["평균"] = df.mean(axis=1)
    df.to_excel(f"{int(step_size/10)}sec_Error.xlsx",float_format="%.6f")
    return df
errors = []
for i in range(len(modelpaths)):
    errors.append(make_df(y_news[i],preds[i],step_sizes[i]))
errors_max = []
for i in range(len(modelpaths)):
    errors_max.append(100-((errors[i].max().max())*100))

step_sizes=[str(size / 10)+"s" for size in step_sizes]
stds = []
def makestd(pred,ydata):
    std = []
    errors = abs(pred - ydata)*100
    std.append(np.std(errors[:,0]))
    std.append(np.std(errors[:,1]))
    std.append(np.std(errors[:,2]))
    std.append(np.std(errors[:,3]))
    std.append(np.std(errors[:,4]))
    std.append(np.std(errors[:,5]))
    return max(std)
for i in range(len(preds)):
    stds.append(makestd(preds[i],y_news[i]))
plt.figure(figsize=(14,6))
plt.bar(step_sizes,errors_max)
plt.ylim(0,100)


plt.errorbar(step_sizes,errors_max,yerr=stds,fmt='none',ecolor='red',capsize=5)
plt.xlabel("Models")
plt.ylabel("Accuracy[%]")
plt.title("Prediction Model Accuracy")
plt.grid()
plt.savefig("bar_error.png")

print()