# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:38:17 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wcat = pd.read_csv("F:\\R\\files\\wc-at.csv")

#to view columns
wcat.columns

#EDA process
￼
#plottings
plt.hist(wcat.Waist)

plt.boxplot(wcat.Waist)

plt.plot(wcat.Waist, wcat.AT, "ro");plt.xlabel("waist");plt.ylabel("tissue")

plt.plot(np.arange(109), wcat.Waist, "-ro")
#correlation
wcat.corr()

wcat.Waist.corr(wcat.AT)

np.corrcoef(wcat.AT, wcat.Waist)

#building a  model
import statsmodels.formula.api as smf

model1 = smf.ols("AT~Waist", data = wcat).fit()

type(model1)

model1.summary()

model1.conf_int(0.05)

pred = model1.predict(wcat)
pred.corr(wcat.AT)
errors = pred - wcat.AT
plt.hist(errors, color= "red")

#exponential model
model2 = smf.ols("AT~np.log(Waist)", data = wcat).fit()

model2.summary()

model2.conf_int(0.05)

pred1 = model2.predict(wcat)
pred.corr(wcat.AT)

#quadratic model
model3 = smf.ols("AT~Waist*Waist*Waist", data = wcat).fit()

model3.summary()

pred2 = model3.predict(wcat)
pred.corr(wcat.AT)

#some basic visualizations

plt.scatter(wcat["Waist"], wcat["AT"], color = "red")

plt.plot(pred,errors, "o");plt.axhline(y=0, color = "green")
np.corrcoef(pred,errors) 

