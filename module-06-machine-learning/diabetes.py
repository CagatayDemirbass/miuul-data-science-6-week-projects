import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.selection import DropCorrelatedFeatures
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import VotingClassifier

def grab_cols(df):
    num_cols = list(df.select_dtypes(include="number"))
    cat_cols = [col for col in df.columns if col not in num_cols]
    num_but_cat = [col for col in num_cols if df[col].nunique()<10]
    cat_but_car = [col for col in cat_cols if df[col].nunique() >20]
    cat_cols = cat_cols + num_but_cat 
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"cat_cols = {len(cat_cols)}")
    print(f"num_cols = {len(num_cols)}")
    print(f"num_but_cat = {len(num_but_cat)}")
    print(f"cat_but_car= {len(cat_but_car)}")
    return cat_cols,num_cols,cat_but_car,num_but_cat

def diabetes_data_prep():
    df = pd.read_csv("C://Users//cagat//Desktop//df//miul//week6//diabetes.csv")
    df.columns = [col.lower() for col in df.columns]
    X = df.drop("outcome",axis=1)
    y =df["outcome"]
    cols=["glucose","bloodpressure","skinthickness","insulin","bmi"]

    for col in cols:
        X.loc[X[col]==0,col] = np.nan
    def outliers(df,variable):
        q1= df[variable].quantile(0.2)
        q3 = df[variable].quantile(0.8)
        iqr = q3 - q1
        lower_lim = q1 - 1.5*iqr
        upper_lim = q3 + 1.5*iqr
        return lower_lim,upper_lim

    def replace_outliers(X,col):
        lower_lim,upper_lim = outliers(X,col)
        X[col].clip(lower=lower_lim,upper=upper_lim,inplace=True)

    replace_outliers(X,"insulin")
    imp_missforest = IterativeImputer(
    estimator=XGBRegressor(n_estimators=300,max_depth=5),
    max_iter=30,
    initial_strategy="median",
    random_state=0
    ).set_output(transform="pandas")

    X=imp_missforest.fit_transform(X)
    def ohe(dataframe,cat_cols):
        dataframe = pd.get_dummies(dataframe,columns=cat_cols,drop_first=True,dtype=int)
        return dataframe
    X["new_glucose_cat"] = pd.cut(x=X["glucose"],bins=[-1,100,140,200],labels=["normal","prediabetes","danger"])

    X.loc[X["age"]<32,"new_age_cat"] = 0
    X.loc[(X["age"]>=32) & (X["age"]<=50),"new_age_cat"]= 1
    X.loc[X["age"]>50,"new_age_cat"] =2

    # X["new_age2"] = pd.cut(x=X["age"],bins=[-1,32,50,100],labels= [0,1,2]) # alt sınıfa dahil eder

    X["new_bmi"] = pd.cut(x=X["bmi"],bins=[-1,18.5,24.9,29.9,100],labels=["underweight","healthy","overweight","obese"])
    X["new_bloodpressure"] = pd.cut(x=X["bloodpressure"],bins=[-1,79,89,123],labels=["normal","hs1","hs2"])
    
    cat_cols,num_cols,cat_but_car,num_but_cat = grab_cols(X)
    X=ohe(X,cat_cols)
    lof = LocalOutlierFactor(n_neighbors=10,n_jobs=-1)
    lof.fit_predict(X)
    X_scores = lof.negative_outlier_factor_
    df = pd.concat([X,y],axis=1)
    df=df.drop(labels =list(df[X_scores<-1.8].index),axis=0 )
    X=df.drop("outcome",axis=1)
    y = df["outcome"]
    sc = StandardScaler().set_output(transform="pandas")
    X = sc.fit_transform(X)
    return X,y




def hyperparameter_optimization(X,y,scoring="roc_auc"):
    rf_params={"max_depth":[3,4,5,6], 
           "min_samples_split":[15,20],
           "n_estimators":[200,300]}

    xgb_params = {"booster":["gblinear","gbtree"],
              "n_estimators":[200,300],
              "reg_lambda":[0.02,0.05],
              "reg_alpha":[0.01,0.02]}

    lr_params = {'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            "max_iter":[5000,1000]}


    classifiers = [("rf",RandomForestClassifier(class_weight='balanced'),rf_params),
               ("xgb",XGBClassifier(objective ="binary:logistic",scale_pos_weight=1.88),xgb_params),
               ("lr",LogisticRegression(solver='liblinear',class_weight='balanced'),lr_params)]
    print("hyperparameter optimization")
    best_models ={}
    for name,classifier,params in classifiers:
        print(f"##### {name}######")
        cv_results = cross_val_score(classifier,X,y,scoring=scoring,cv=10,n_jobs=-1).mean()
        print(f"{scoring} (Before): {cv_results}")
        
        gs = GridSearchCV(classifier,params,cv=10,scoring=scoring).fit(X,y)
        final_model = classifier.set_params(**gs.best_params_)
        
        cv_results = cross_val_score(final_model,X,y,scoring=scoring,cv=10,n_jobs=-1).mean()
        print(f"{scoring} (After): {cv_results}")
        print(f"{name} best_params: {gs.best_params_}", end="//n//n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models,X,y):
    voting_clf=VotingClassifier(estimators = [("lr",best_models["lr"]),
                                            ("rf",best_models["rf"]),
                                            ("xg",best_models["xgb"])],
                              voting='soft',
                            weights=[1,1,1])
    cv_results = cross_validate(voting_clf,X,y,cv=10,scoring=["accuracy","roc_auc","recall"])
    print(f"accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"recall: {cv_results['test_recall'].mean()}")
    print(f"roc_auc: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

def main():
    X,y = diabetes_data_prep()
    best_models = hyperparameter_optimization(X,y,scoring="roc_auc")
    voting_clf = voting_classifier(best_models,X,y)
    joblib.dump(voting_clf,"voting_clf.pkl")
    return voting_clf

if __name__=="__main__":
    main()