#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[3]:


def load_financial_data(start_date, end_date, output_file):
    try:
        df = pd.read_csv(output_file)
        print('File data found...reading APPL data')
    except FileNotFoundError:
        print('File not found...downloading the APPL data')
        df = data.DataReader('GOOG', 'yahoo', start_date, end_date)
        df.to_csv(output_file)

    return df


# In[4]:


df_APPL = load_financial_data( start_date='2001-01-01', end_date='2018-01-01', output_file='df_APPL.csv')


# In[5]:


df_APPL.head()


# In[74]:


df1=df_APPL.reset_index()['Close']


# In[75]:


df1.shape


# In[76]:


plt.plot(df1)


# In[77]:


scaler =MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[78]:


df1.shape


# In[79]:


df1


# In[80]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[81]:


training_size,test_size


# In[82]:


len(train_data),len(test_data)


# In[83]:


print(train_data)


# In[84]:


print(test_data)


# In[85]:


def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)


# In[86]:


time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)


# In[87]:


print(X_train.shape),print(y_train.shape)


# In[88]:


print(X_test.shape),print(y_test.shape)


# In[89]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[90]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# In[91]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[92]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=2)


# In[93]:


model.summary()


# In[94]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[95]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[96]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[97]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[98]:


y_test_inverse=scaler.inverse_transform(y_test.reshape(-1,1))


# In[103]:


plt.figure(figsize=(12,6))
plt.plot(y_test_inverse,'g',label='GOOG Returns')
plt.plot(test_predict,'r',label='Strategy Returns')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[31]:


look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[43]:


hundays=len(test_data)-100


# In[44]:


x_input=test_data[hundays:].reshape(1,-1)
x_input.shape


# In[45]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[46]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[47]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[48]:


len(df1)


# In[51]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[5184:])


# In[55]:


df3=scaler.inverse_transform(df3).tolist()


# In[56]:


plt.plot(df3)


# In[ ]:




