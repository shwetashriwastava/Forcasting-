#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf 


# In[4]:


Air = pd.read_excel("D:\\data science\\assignments\\ass-18 forecasting\\Airlines+Data.xlsx")


# In[5]:


Air


# In[6]:


Air.head()


# In[7]:


Air.info


# In[8]:


Air.shape


# In[9]:


plt.title("Line Plot", size = 15, weight = 'bold')
plt.ylabel("Passengers", size = 10, weight = 'bold')
plt.plot(Air['Passengers'])


# In[11]:


Air["month"] = Air.Month.dt.strftime("%b")


# In[12]:


Air


# In[13]:


data = pd.get_dummies(Air["month"])


# In[14]:


data


# In[16]:


Air1 = pd.concat([Air,data],axis=1)


# In[17]:


Air1["t"] = np.arange(1,97)
Air1["t_squared"] = Air1["t"]*Air1["t"]
Air1.columns
Air1["log_passengers"] = np.log(Air1["Passengers"])


# In[19]:


Air1


# In[20]:


train= Air1.head(88)
test=Air1.tail(8)
Air1.Passengers.plot()


# In[21]:


indexedDataset = Air1.set_index(['Month'])
indexedDataset.head(5)


# In[22]:


Air1.Passengers.plot(label="org")
for i in range(2,10,2):
    Air1["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[24]:


import statsmodels.formula.api as smf


# In[25]:


#linear model
linear= smf.ols('Passengers~t',data=Air1).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Passengers'])-np.array(predlin))**2))
print("Root Mean Square Error : ",rmselin)


# In[26]:


#quadratic model
quad=smf.ols('Passengers~t+t_squared',data=Air1).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_squared']])))
rmsequad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predquad))**2))
print("Root Mean Square Error : ",rmsequad)


# In[27]:


#exponential model
expo=smf.ols('log_passengers~t',data=Air1).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predexp)))**2))
print("Root Mean Square Error : ",rmseexpo)


# In[29]:


#additive seasonality
additive= smf.ols('Passengers~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predadd))**2))
print("Root Mean Square Error : ",rmseadd)


# In[30]:


#additive seasonality with linear trend
addlinear= smf.ols('Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
predaddlinear


# In[31]:


rmseaddlinear=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddlinear))**2))
print("Root Mean Square Error : ",rmseaddlinear)


# In[32]:


#additive seasonality with quadratic trend
addquad=smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddquad))**2))
print("Root Mean Square Error : ",rmseaddquad)


# In[33]:



#multiplicative seasonality
mulsea=smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmsemul= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmul)))**2))
print("Root Mean Square Error : ",rmsemul)


# In[34]:



#multiplicative seasonality
mulsea=smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmsemul= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmul)))**2))
print("Root Mean Square Error : ",rmsemul)


#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_mul_quad)))**2))
print("Root Mean Square Error : ",rmse_mul_quad)


# In[35]:



data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsequad])}


# In[36]:


Rmse=pd.DataFrame(data)
Rmse


# In[38]:


data_Predict = pd.read_excel("D:\\data science\\assignments\\ass-18 forecasting\\Airlines+Data.xlsx")


# In[39]:


Final_pred = smf.ols('log_passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Air1).fit()


# In[40]:


pred_new  = pd.Series(Final_pred.predict(Air1))


# In[41]:


pred_new


# # 2nd data set

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf 


# In[43]:


coca=pd.read_excel("D:\\data science\\assignments\\ass-18 forecasting\\CocaCola_Sales_Rawdata.xlsx")


# In[44]:


coca.head()


# In[46]:


coca.info


# In[47]:


coca.shape


# In[48]:


plt.title("Line Plot", size = 15, weight = 'bold')
plt.ylabel("Sales", size = 10, weight = 'bold')
plt.plot(coca['Sales'])


# In[49]:


quarters = ['Q1','Q2','Q3','Q4']
n = coca['Quarter'][0]
n[0:2]


# In[52]:


coca['quarter'] = 0
for i in range(42):
    n = coca['Quarter'][i]
    coca['quarter'][i] = n[0:2]


# In[54]:


dummy = pd.DataFrame(pd.get_dummies(coca['quarter']))


# In[55]:


dummy


# In[58]:


coca_c=pd.concat([coca,dummy],axis=1)


# In[59]:


coca_c


# In[60]:


coca_c["t"] = np.arange(1,43)

coca_c["t_squared"] = coca_c["t"]*coca_c["t"]
coca_c.columns
coca_c["log_Sales"] = np.log(coca_c["Sales"])


# In[61]:


coca_c


# In[62]:


plt.figure(figsize=(12,8))
heatmap_y_quarter = pd.pivot_table(data=coca_c,values="Sales",index="Quarter",columns="quarter",aggfunc="mean", fill_value = 0)
sns.heatmap(heatmap_y_quarter,annot=True,fmt="g") #fmt is format of the grid values


# In[63]:


# Boxplot for ever
plt.figure(figsize=(8,6))
sns.boxplot(x="quarter",y="Sales",data=coca_c)


# In[64]:


lag_plot(coca_c['Sales'])
plt.title("Lag Plot", size = 15, weight = "bold")
plt.show()


# In[65]:


plot_acf(coca_c['Sales'], lags = 30, color = 'red')               # lags = 30 means it will plot for k = 30 lags 
plt.xlabel("No of lags, k = 30", size = 10, weight = 'bold')
plt.ylabel("Autocorrelation (r2 value)", size = 20, weight = 'bold')
plt.show()


# In[66]:



plt.figure(figsize=(12,3))
sns.lineplot(x="quarter",y="Sales",data=coca_c)


# In[68]:


train =coca_c.head(37)
test  =coca_c.tail(4)


# In[69]:


train


# In[70]:


test


# In[71]:


import statsmodels.formula.api as smf 


# In[72]:



linear = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[73]:


Quad = smf.ols('Sales~t+t_squared',data=train).fit() #quadratic model
pred_Quad = pd.Series(Quad.predict(test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_Quad))**2))
print("Root Mean Square Error : ",rmse_Quad)


# In[74]:



Exp = smf.ols('log_Sales~t',data=train).fit() #exponential model
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Exp)))**2))
print("Root Mean Square Error : ",rmse_Exp)


# # additive seasonality with linear treand
# 

# In[76]:


add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=train).fit() #additive seasonality model
pred_add_sea = pd.Series(add_sea.predict(test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
print("Root Mean Square Error : ",rmse_add_sea)


# # additive seasonality with quadratic trend
# 

# In[77]:


add_sea_quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=train).fit() #additive seasonality qudratic model
pred_add_sea_quad = pd.Series(add_sea_quad.predict(test[['t','t_squared','Q1','Q2','Q3','Q4']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
print("Root Mean Square Error : ",rmse_add_sea_quad)


# In[78]:


#multiplicative seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = train).fit() #multiplicative seasonality model
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
print("Root Mean Square Error : ",rmse_Mult_sea)


# In[79]:


#multiplicative additive seasonality
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = train).fit() #multiplicative additive seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
print("Root Mean Square Error : ",rmse_Mult_add_sea) 


# In[80]:


#tabuling rmes value
Final_data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
Final_result = pd.DataFrame(Final_data) #data frame of final result
Final_result.sort_values(['RMSE_Values'])


# In[81]:


data_Predict = pd.read_excel("D:\\data science\\assignments\\ass-18 forecasting\\CocaCola_Sales_Rawdata.xlsx")


# In[83]:


Final_pred = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=coca_c).fit()


# In[85]:


pred_new  = pd.Series(Final_pred.predict(coca_c))


# In[86]:


pred_new


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




