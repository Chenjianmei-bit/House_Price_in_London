# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import scipy
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import mixture
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#%% load data  

hp=pd.read_csv("London_cleaned_20201208_2.csv")

#%% Class for Best SVR

class pricePredictor():
    
    def __init__(self,data,responseCol=0,preditorColStart=1,preditorColEnd=68):
        self.data=data
        self.responseCol=responseCol
        self.preditorColStart=preditorColStart
        self.preditorColEnd=preditorColEnd
            
    def perfectVariable(self,xTrain,xTest,
                        scalerOrNot=True,
                        pcaOrNot=True):
        # Convert X to avoid missingvalue and then scaler,PCA
        # C means converter
        # missing value
        impC = SimpleImputer()   
        impC.fit(xTrain)
        xTrain = impC.transform(xTrain)
        xTest = impC.transform(xTest)
        # scaler
        if scalerOrNot:
            scalerC = preprocessing.StandardScaler()
            scalerC.fit(xTrain)
            xTrain = scalerC.transform(xTrain)
            xTest = scalerC.transform(xTest)
        if pcaOrNot:
            pcaC = PCA(0.95)
            pcaC.fit(xTrain,xTest)
            xTrain = pcaC.transform(xTrain)
            xTest = pcaC.transform(xTest)
        rs={
            "xTrainConverted":xTrain,
            "xTestConverted":xTest
            }
        return(rs)        
    
    def segementSet(self,testRatio=0.2,bootstrap=True,bootstrapMultiple=3,
                    scalerOrNot=True,pcaOrNot=True,newData=False,dtN=None):
        ixAll=np.array(np.array(range(self.data.shape[0])))
        testCount=int(self.data.shape[0]*testRatio)
        testIx=np.random.choice(ixAll,
                                testCount,
                                replace=False)
        if bootstrap:
            trainIx=np.random.choice(ixAll[~np.isin(ixAll,testIx)],
                                     (self.data.shape[0]-testCount)*bootstrapMultiple,
                                     replace=True)
        else:
            trainIx=ixAll[~np.isin(ixAll,testIx)]
        if newData:
            xA=self.data.iloc[:,self.preditorColStart:self.preditorColEnd].values
            yA=self.data.iloc[:,self.responseCol].values
            xE=dtN.iloc[:,self.preditorColStart:self.preditorColEnd].values
            yE=dtN.iloc[:,self.responseCol].values
        else:
            xA=self.data.iloc[trainIx,self.preditorColStart:self.preditorColEnd].values
            yA=self.data.iloc[trainIx,self.responseCol].values
            xE=self.data.iloc[testIx,self.preditorColStart:self.preditorColEnd].values
            yE=self.data.iloc[testIx,self.responseCol].values
            # convert X for train and X for test
        variableConverted=self.perfectVariable(xA,xE,
                                               scalerOrNot=scalerOrNot,
                                               pcaOrNot=pcaOrNot)
        xA=variableConverted['xTrainConverted']
        xE=variableConverted['xTestConverted']
        # result
        aeSet={
            "xA":xA,
            "yA":yA,
            "xE":xE,
            "yE":yE
            }
        return(aeSet)
    def getMAPE(self,prediction,actual):
        prediction=np.array(prediction)
        actual=np.array(actual)
        MAPE=np.mean(np.abs(prediction - actual)/np.abs(actual))
        return(MAPE)
    
    def fitMod(self,method,
               testRatio=0.2,bootstrap=True,
               bootstrapMultiple=3,
               B=10,plot=True,
               scalerOrNot=True,pcaOrNot=True):
        MAPEList=np.array([])
        MAEList=np.array([])
        RMSEList=np.array([])
        for i in range(B):
            aeSet=self.segementSet(testRatio=testRatio,
                                   bootstrap=bootstrap,
                                   bootstrapMultiple=bootstrapMultiple,
                                   scalerOrNot=scalerOrNot,
                                   pcaOrNot=pcaOrNot)
            if method=="LinearRegression":
                mod=LinearRegression()
            elif method=="DecisionTreeRegressor":
                mod=DecisionTreeRegressor()
            elif method=="DecisionTreeRegressor":
                mod=DecisionTreeRegressor()
            elif method=="RandomForestRegressor":
                mod=RandomForestRegressor()
            elif method=="SVR":
                mod=SVR()
            elif method=="Bayes":
                mod=BayesianRidge()
            # fit model
            mod.fit(aeSet['xA'],aeSet['yA'])
            # predict
            prd=mod.predict(aeSet['xE'])
            # calculate MAE and RMSE
            MAPE=self.getMAPE(aeSet['yE'],prd)
            MAE=mean_absolute_error(aeSet['yE'],prd)
            RMSE=mean_squared_error(aeSet['yE'],prd, squared=False)
            # record metrics for each loop
            MAPEList=np.append(MAPEList,MAPE)
            MAEList=np.append(MAEList,MAE)
            RMSEList=np.append(RMSEList,RMSE)
        if plot:
            plt.figure()
            figMAPE=sns.kdeplot(MAPEList).set(title=('MAPE of '+method),
                                              xlim=(0, 1))
            plt.show()
            plt.figure()
            figMAE=sns.kdeplot(MAEList).set(title=('MAE of '+method))
            plt.show()
            plt.figure()
            figRMSE=sns.kdeplot(RMSEList).set(title=('RMSE of '+method))
            plt.show()
            print("Mean of MAPE of ",method," is ",str(np.mean(MAPEList)))
            print("Variance of MAPE of ",method," is ",str(np.var(MAPEList)))
            print("Mean of MAE of ",method," is ",str(np.mean(MAEList)))
            print("Variance of MAE of ",method," is ",str(np.var(MAEList)))
            print("Mean of RMSE of ",method," is ",str(np.mean(RMSEList)))
            print("Variance of RMSE of ",method," is ",str(np.var(RMSEList)))
            return(np.mean(MAEList))
        else:
            print("Mean of MAPE of ",method," is ",str(np.mean(MAPEList)))
            print("Variance of MAPE of ",method," is ",str(np.var(MAPEList)))
            print("Mean of MAE of ",method," is ",str(np.mean(MAEList)))
            print("Variance of MAE of ",method," is ",str(np.var(MAEList)))
            print("Mean of RMSE of ",method," is ",str(np.mean(RMSEList)))
            print("Variance of RMSE of ",method," is ",str(np.var(RMSEList)))
            return(np.mean(MAEList))
        
    def forecastNew(self,method,newFileName,
               testRatio=0.05,bootstrap=True,
               bootstrapMultiple=3,
               scalerOrNot=True,pcaOrNot=True):
        dtN=pd.read_csv(newFileName)
        aeSet=self.segementSet(testRatio=testRatio,
                               bootstrap=bootstrap,
                               bootstrapMultiple=bootstrapMultiple,
                               scalerOrNot=scalerOrNot,
                               pcaOrNot=pcaOrNot,
                               newData=True,dtN=dtN)
        if method=="LinearRegression":
            mod=LinearRegression()
        elif method=="DecisionTreeRegressor":
            mod=DecisionTreeRegressor()
        elif method=="DecisionTreeRegressor":
            mod=DecisionTreeRegressor()
        elif method=="RandomForestRegressor":
            mod=RandomForestRegressor()
        elif method=="SVR":
            mod=SVR()
        elif method=="Bayes":
            mod=BayesianRidge()
        # fit model
        mod.fit(aeSet['xA'],aeSet['yA'])
        # predict
        prd=mod.predict(aeSet['xE'])
        # calculate MAE and RMSE
        MAPE=self.getMAPE(aeSet['yE'],prd)
        MAE=mean_absolute_error(aeSet['yE'],prd)
        RMSE=mean_squared_error(aeSet['yE'],prd, squared=False)
        print("MAPE is ",str(MAPE))
        print("MAE is ",str(MAE))
        print("RMSE is ",str(RMSE))
        return(prd)

#%% Example

# Compare model
methodList=["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "SVR", "Bayes"]
pp=pricePredictor(hp)
for method in methodList:
    print(method)
    pp.fitMod(method=method,
              B=10,
              bootstrap=False)

# Predict for new data  
newFileName="London_new_20201208.csv"
pp=pricePredictor(hp)
prd=pp.forecastNew("RandomForestRegressor",
                   newFileName)