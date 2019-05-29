import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import random
import time


np.random.seed(7)
input_file="2018-69-12.csv"
df = read_csv(input_file, header=None, index_col=None, delimiter=',')
all_y = df[0].values   # take close price column[5]
dataset = all_y.reshape(-1, 1) # 将一维数组，转化为2维数组
dataset = dataset.astype('float32') # 转化为32位浮点数，防止0数据

scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化
dataset = scaler.fit_transform(dataset)


def create_dataset(dataset, look_back=1): # convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back -10 ):  # 后一个数据与前look_back个数据相关
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[(i + look_back): i + look_back + 10, 0])
    return np.array(dataX), np.array(dataY) # 生成输入数据和输出数据


def create_datasetP(dataset, look_back=1):
    dataP = []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), 0]
        dataP.append(a)
    return np.array(dataP)


def r2_score(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()


def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp
        mape = sums * (100 / num)

    return mape


look_back =168   # timestep预测下一步，需要之前的timestep个？？？

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, look_back)))# n_steps,n_features
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))

model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))

model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.compile(loss='mse', optimizer='adam')
# 随机n次  性能度量：评估模型的泛化能力，比如准确率/召回率/
# 连续数据的预测，常用 mean_absolute_error,绝对值导致函数不光滑，某些点处不可求导。
# 而l2范数的mean_squared_error可求。 RMSE虽然广为使用，但是其存在一些缺点，因为它是使用平均误差，而平均值对异常点（outliers）较敏感，如果回归器对某个点的回归值很不理性，那么它的误差则较大，从而会对RMSE的值有较大影响，即平均值是非鲁棒的。
# r2_score   在特征值多的情况下适用

# global l2, l3, l4
# l1, = plt.plot(scaler.inverse_transform(dataset), color='blue', linewidth=1.5)
# plt.plot(scaler.inverse_transform(dataset), color='blue', linewidth=1.5)
for i in range(800):
    print(" 当前回合：", i)
    start = random.randint(0, 2600)
    datasetX, datasetY = create_dataset(dataset[start:(start + 179), :], look_back)
    # reshape input to be [samples, time steps, features]转化数据维度
    datasetX = np.reshape(datasetX, (datasetX.shape[0], 1, datasetX.shape[1]))
    model.fit(datasetX, datasetY, epochs=800, batch_size=8, verbose=1)

    datasetPredict = model.predict(datasetX)
    datasetPredict = scaler.inverse_transform(datasetPredict)
    datasetY = scaler.inverse_transform(datasetY)

    trainScore = MAPE(datasetY[0, :], datasetPredict[0, :])
    print('mapeTRAIN:%f%%' % trainScore)

    datasetPredictPlot = np.empty_like(dataset)
    datasetPredictPlot[:, :] = np.nan
    # for i in range(10):
    #     datasetPredictPlot[look_back + start:len(datasetPredict[0, :]) + look_back + start, 0] = datasetPredict[0, :]
    #     plt.plot(datasetPredictPlot, color='red', linewidth=1.0)

    # if i == 7:
    #     datasetPredictPlot[look_back + start:len(datasetPredict[0, :]) + look_back + start, 0] = datasetPredict[0, :]
    #     l2, = plt.plot(datasetPredictPlot, color='red', linewidth=1.0)
    # if i == 8:
    #     datasetPredictPlot[look_back + start:len(datasetPredict[0, :]) + look_back + start, 0] = datasetPredict[0, :]
    #     l3, = plt.plot(datasetPredictPlot, color='green', linewidth=1.0)
    # if i == 9:
    #     datasetPredictPlot[look_back + start:len(datasetPredict[0, :]) + look_back + start, 0] = datasetPredict[0, :]
    #     l4, = plt.plot(datasetPredictPlot, color='yellow', linewidth=1.0)
    # print(datasetPredictPlot[:, 0])

# plt.legend([l1, l2, l3, l4], ('raw-data1', 'train-data2', 'train-data3', 'train-data4'), loc='best')


prev_seq = dataset[-look_back:]
prev_seqX = create_datasetP(prev_seq, look_back)
prev_seqX = np.reshape(prev_seqX, (prev_seqX.shape[0], 1, prev_seqX.shape[1]))
predict = model.predict(prev_seqX)
predict = scaler.inverse_transform(predict)
#对比真实数据
test_file = "10.csv"
dfp = read_csv(test_file, header=None, index_col=None, delimiter=',')
df_p = dfp[0].values
predict24 = df_p.reshape(-1, 1) # 将一维数组，转化为2维数组
predict24 = predict24.astype('float32') # 转化为32位浮点数，防止0数据

print(len(predict24[:, 0]))

testScore = MAPE(predict24[:, 0], predict[0, :])

print('mape10TEST:%f%%' % testScore)


df = pd.DataFrame(data={"predict24": np.around(list(predict24.reshape(-1)), decimals=2), "preidct": np.around(list(predict.reshape(-1)), decimals=2)})
df.to_csv("lstm_result.csv", sep=';', index=None)
# plt.plot(list(range(len(dataset), len(dataset) + len(predict))), predict, color='r')
plt.plot(range(len(predict[0, :])), predict[0, :], color='r')
plt.plot(range(len(predict24)), predict24, color='b')

plt.show()









