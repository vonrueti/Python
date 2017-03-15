6# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


"""
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

df_1 = pd.read_csv("H:/DATA/Python/Titanic/train.csv")

"""
###################################

Part one includes the Data Preparation

###################################
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Extract the Titel from the Name~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
df_1["Titel"]=df_1.Name.str.split(",").str[1]
df_1["Titel"]=df_1.Titel.str.split(".").str[0]
df_1["Titel"]=df_1.Titel.str.strip()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Separate Betweeen Nan and not nan~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
df_10= df_1[pd.isnull(df_1['Age'])]
df_11= df_1[pd.notnull(df_1['Age'])]
            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Titel analysis and reformulation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#           
pd.crosstab(df_1['Survived'], df_1['Titel'],normalize='columns',margins=True)            
pd.crosstab(df_1['Survived'], df_1['Titel'],margins=True)
pd.crosstab(df_10['Survived'], df_10['Titel'],normalize='columns',margins=True)            
pd.crosstab(df_10['Survived'], df_10['Titel'],margins=True)   
pd.crosstab(df_1['Sex'], df_1['Titel'],margins=True)
sns.boxplot(x="Titel", y="Age",data=df_1);

df_1.loc[(df_1["Titel"].isin(["Mme","Ms","Lady","the Countess"]) | ((df_1["Titel"]=="Dr") & (df_1["Sex"]=="female"))),"Titel"]='Mrs'
df_1.loc[(df_1["Titel"].isin(["Mlle"])),"Titel"]='Miss'
df_1.loc[(df_1["Titel"].isin(["Don","Sir","Jonkheer","Capt","Col","Major","Rev"]) | ((df_1["Titel"]=="Dr") & (df_1["Sex"]=="male"))),"Titel"]='Mr'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Get the number of Person for each Ticket an calculate Fare per Person (AFare) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
a = df_1.groupby('Ticket',as_index=False).count()
a = a[["Ticket","Fare"]]
a = a.rename(columns={"Fare": "NPerson"})

df_2 = df_1.set_index('Ticket').join(a.set_index('Ticket'))
df_2["AFare"]= df_2["Fare"] / df_2["NPerson"] #AFare= Fare Per Person

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculate If person travels alone (Alone=1 when NPerson=1, Alone=0 when NPerson >1)
df_2["Alone"]=1
df_2.loc[(df_2["NPerson"]>1),"Alone"]=0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calculate for each "Family Indicator" a Boolean, when there is a Family Member then 1 else e (for Parch and SibSp). When the boolean of Parch or SibSp is one then Family=1 else 0
df_2["ParCh_B"]=1
df_2["SibSp_B"]=1
df_2.loc[(df_2["Parch"]<1),"ParCh_B"]=0
df_2.loc[(df_2["SibSp"]<1),"SibSp_B"]=0
df_2["Family"]=0
df_2.loc[(df_2["SibSp_B"]==1) | (df_2["ParCh_B"]==1),"Family"]=1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Transform Variables to other types        
df_2["Family"]=df_2["Family"].astype('category')
df_2["Alone"]=df_2["Alone"].astype('category')
df_2["Titel"]=df_2["Titel"].astype('category')
df_2["ParCh_B"]=df_2["ParCh_B"].astype('category')
df_2["SibSp_B"]=df_2["SibSp_B"].astype('category')

df_2["Sex"]=df_2["Sex"].astype('category')
df_2["Survived"]=df_2["Survived"].astype('category')
df_2["survived"]=df_2["Survived"].astype('int64')
df_2["Pclass"]=df_2["Pclass"].astype('category')
df_2["Embarked"]=df_2["Embarked"].astype('category')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Group Age into a new Group
df_2["Age_G"]=pd.cut(df_2["Age"], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120], labels=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120])
df_2["Age_E"]=pd.cut(df_2["Age"], [0, 15, 20,120], labels=[15, 20, 120])
df_2["Age_Er"]=pd.cut(df_2["Age"], [0, 15,120], labels=[15, 120])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Set for the Missing Values an Average Value
df_2.groupby(['Age_Er']).mean()
pd.crosstab(df_2['Age_Er'], df_2['SibSp'],margins=True)
df_2.loc[(((((df_2['Titel']=='Miss') & (df_2['SibSp']> 2)) | (df_2['Titel']=='Master'))==1 ) & (pd.isnull(df_2['Age']))),"Age"]=6
df_2.loc[(((((df_2['Titel']=='Miss') & (df_2['SibSp']> 2)) | (df_2['Titel']=='Master'))==0 ) & (pd.isnull(df_2['Age']))),"Age"]=33
df_2["Age_G"]=pd.cut(df_2["Age"], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120], labels=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120])
df_2["Age_E"]=pd.cut(df_2["Age"], [0, 15, 20,120], labels=[15, 20, 120])
df_2["Age_Er"]=pd.cut(df_2["Age"], [0, 15,120], labels=[15, 120])

df_2.dtypes

"""
###################################

Part two includes the data analysis

Following Variables need to investigated:
Pclass, Fare, AFare
Sex, Age_G, Age_E, Age_Er, Age, Titel
SibSp, Parch, NPerson, Alone, ParCh_B, SibSp_B, Family
Embarked

Against the Variable survived/Survived

Not used: PassengerId, Cabin, Name(indirect Titel)
###################################
"""

"""
Analysis of Pclass, Fare, AFare
"""
mosaic(df_2, ['Pclass', 'Survived']);
pd.crosstab(df_2['Survived'], df_2['Pclass'],normalize='columns',margins=True)
#====>           Pclass seems to have an Influence

sns.boxplot(x="Survived", y="Fare",data=df_2);
sns.boxplot(x="Survived", y="AFare",data=df_2);
#====>          There seems to be a difference, but the question is, if this variable is necessary (maybe we can just use Pclass)

sns.boxplot(x="Pclass", y="AFare",data=df_2);
sns.boxplot(x="Pclass", y="AFare",data=df_2);
#====>          The information of AFare seems to be included in Pclass


#Might be enough to just include Pclass (else AFare)

"""
Sex, Age_G, Age_E, Age_Er, Age, Titel
"""
mosaic(df_2, ['Sex', 'Survived']);
pd.crosstab(df_2['Survived'], df_2['Sex'],normalize='columns',margins=True)
#====>          Female have a higher prob. to survive

sns.barplot(x="Age_G", y="survived", data=df_2);
sns.barplot(x="Age_E", y="survived", data=df_2);
sns.barplot(x="Age_Er", y="survived", data=df_2);
#====>          It makes sense to group the age (adults/childrean)


mosaic(df_2, ['Titel', 'Survived']);
pd.crosstab(df_2['Survived'], df_2['Titel'],normalize='columns',margins=True)

pd.crosstab(df_2['Titel'], df_2['Age_Er'],normalize='columns',margins=True)
#====>          In  Titel we have the age and sex information, maybe this variable is enough


#====>          Include Titel, maybe Sex / Age_Er

"""
SibSp, Parch, NPerson, Alone, ParCh_B, SibSp_B, Family
"""
sns.barplot(x="SibSp", y="survived", data=df_2);
sns.barplot(x="Parch", y="survived", data=df_2);
pd.crosstab(df_2['Survived'], df_2['SibSp'],margins=True)
pd.crosstab(df_2['Survived'], df_2['Parch'],margins=True)

sns.barplot(x="ParCh_B", y="survived", data=df_2);
sns.barplot(x="SibSp_B", y="survived", data=df_2);
sns.barplot(x="Family", y="survived", data=df_2);

sns.barplot(x="NPerson", y="survived", data=df_2);
sns.barplot(x="Alone", y="survived", data=df_2);

#====> I would tak alone and family into the model (or maybe family separated by sibsp and parch)
"""
Embarked
"""
sns.barplot(x="Embarked", y="survived", data=df_2);
pd.crosstab(df_2['Embarked'], df_2['Titel'],normalize='columns',margins=True)
pd.crosstab(df_2['Pclass'], df_2['Embarked'],margins=True)
sns.barplot(x="Embarked", y="Titel", data=df_2);
sns.barplot(x="Embarked", y="Titel", data=df_2);

#===>   In my opinion Embarked could lead to a missclassification

"""
###################################

Part three which Variable should we take into account?

###################################
"""
#Might be enough to just include Pclass (else AFare)
#Include Titel, maybe Sex / Age_Er
#I would tak alone and family into the model (or maybe family separated by sibsp and parch)
#In my opinion Embarked could lead to a missclassification
df_2.dtypes
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


y, X = dmatrices('survived ~ Parch + SibSp +C(Alone) + C(Family) + C(ParCh_B) + C(SibSp_B) +  C(Pclass)+ AFare + C(Titel) + C(Sex) + C(Age_Er) +C(Embarked) ',
                  df_2, return_type="dataframe")

X.dtypes 
X = X.rename(columns = {'C(Age_Er)[T.120]':'adult',
                        'C(Embarked)[T.Q]':'Queenstown',
                        'C(Embarked)[T.S]':'Southampton',
                        'C(Pclass)[T.2]':'midClass',
                        'C(Pclass)[T.3]':'lowClass',
                        'C(Sex)[T.male]':'male'})
y = np.ravel(y)

DTree = tree.DecisionTreeClassifier()
DTree = DTree.fit(X, y)
names = list(X)
sorted(zip(map(lambda x: round(x, 4), DTree.feature_importances_), names), reverse=True)
"""
###################################

Part four Run Classification Models

###################################
"""

y, X = dmatrices('survived ~ Parch + SibSp + C(Pclass) + C(Titel) +C(Alone)',
                  df_2, return_type="dataframe")

X.dtypes 
X = X.rename(columns = {'C(Age_Er)[T.120]':'adult',
                        'C(Embarked)[T.Q]':'Queenstown',
                        'C(Embarked)[T.S]':'Southampton',
                        'C(Pclass)[T.2]':'midClass',
                        'C(Pclass)[T.3]':'lowClass',
                        'C(Sex)[T.male]':'male'})
y = np.ravel(y)

LR_Model = LogisticRegression()
LR_Model = LR_Model.fit(X, y)
LR_Model.score(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
LR_Model_2 = LogisticRegression()
LR_Model_2 = LR_Model_2.fit(X_train, y_train)
LR_Model_2.score(X_test, y_test)

LR_Model_scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
LR_Model_scores.mean()


svm_Model = svm.SVC()
svm_Model= svm_Model.fit(X, y)
svm_Model_pre = svm_Model.predict(X)
sum(svm_Model_pre==y)/len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
svm_Model_2 = svm.SVC()
svm_Model_2 = svm_Model_2.fit(X_train, y_train)
svm_Model_pre_2 = svm_Model_2.predict(X_test)
sum(svm_Model_pre_2==y_test)/len(y_test)

svm_Model_scores = cross_val_score(svm.SVC(), X, y, scoring='accuracy', cv=10)
svm_Model_scores.mean()
svm_Model_scores.var()

DTree = tree.DecisionTreeClassifier()
DTree = DTree.fit(X, y)
DTree_Pre = DTree.predict(X)
sum(DTree_Pre==y)/len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
DTree_2 = tree.DecisionTreeClassifier()
DTree_2 = DTree_2.fit(X_train, y_train)
DTree_pre_2 = DTree_2.predict(X_test)
sum(DTree_pre_2==y_test)/len(y_test)

DTree_scores = cross_val_score(tree.DecisionTreeClassifier(), X, y, scoring='accuracy', cv=10)
DTree_scores.mean()
DTree_scores.var()

#I should also have a look on some figures, but I don't think, that I'll have more time.
"""
###################################

Part five Run the best Model on the test Data set.

###################################
"""

df_3 = pd.read_csv("H:/DATA/Python/Titanic/test.csv")


df_3["Titel"]=df_3.Name.str.split(",").str[1]
df_3["Titel"]=df_3.Titel.str.split(".").str[0]
df_3["Titel"]=df_3.Titel.str.strip()            
            
pd.crosstab(df_3['Sex'], df_3['Titel'],margins=True)
sns.boxplot(x="Titel", y="Age",data=df_3);

df_3.loc[(df_3["Titel"].isin(["Mme","Ms","Lady","the Countess","Dona"]) | ((df_3["Titel"]=="Dr") & (df_3["Sex"]=="female"))),"Titel"]='Mrs'
df_3.loc[(df_3["Titel"].isin(["Mlle"])),"Titel"]='Miss'
df_3.loc[(df_3["Titel"].isin(["Don","Sir","Jonkheer","Capt","Col","Major","Rev"]) | ((df_3["Titel"]=="Dr") & (df_3["Sex"]=="male"))),"Titel"]='Mr'

a = df_3.groupby('Ticket',as_index=False).count()
a = a[["Ticket","Fare"]]
a = a.rename(columns={"Fare": "NPerson"})

df_4 = df_3.set_index('Ticket').join(a.set_index('Ticket'))

df_4["AFare"]= df_4["Fare"] / df_4["NPerson"]
df_4["Alone"]=1
df_4.loc[(df_4["NPerson"]>1),"Alone"]=0
df_4["ParCh_B"]=1
df_4["SibSp_B"]=1
df_4.loc[(df_4["Parch"]<1),"ParCh_B"]=0
df_4.loc[(df_4["SibSp"]<1),"SibSp_B"]=0
         
df_4["NPerson_C"]=df_4["NPerson"].astype('category')
df_4["Alone"]=df_4["Alone"].astype('category')
df_4["Titel_C"]=df_4["Titel"].astype('category')
df_4["ParCh_B"]=df_4["ParCh_B"].astype('category')
df_4["SibSp_B"]=df_4["SibSp_B"].astype('category')

df_4["Sex"]=df_4["Sex"].astype('category')

df_4["survived"]=1
df_4["Pclass"]=df_4["Pclass"].astype('category')
df_4["Embarked"]=df_4["Embarked"].astype('category')
df_4["SibSp_C"]=df_4["SibSp"].astype('category')
df_4["Parch_C"]=df_4["Parch"].astype('category')
df_4["Titel_C"]=df_4["Titel"].astype('category')
df_4["Age_G"]=pd.cut(df_4["Age"], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120], labels=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120])
df_4["Age_E"]=pd.cut(df_4["Age"], [0, 15, 20,120], labels=[15, 20, 120])
df_4["Age_Er"]=pd.cut(df_4["Age"], [0, 15,120], labels=[15, 120])
df_4.dtypes




t, Xtest = dmatrices('survived ~ Parch + SibSp + C(Pclass) + C(Titel) +C(Alone)',
                  df_2, return_type="dataframe")


Xtest = Xtest.rename(columns = {'C(Pclass)[T.2]':'midClass',
                        'C(Pclass)[T.3]':'lowClass',
                        'C(Sex)[T.male]':'male',
                        'C(Age_Er)[T.120]':'Erwachsen'})

yPredict = svm_Model.predict(Xtest)
myResult =DataFrame()
myResult["PassengerId"] = df_4["PassengerId"]
myResult["Survived"] = yPredict
myResult.dtypes
myResult["Survived"]=myResult["Survived"].astype('int64')
myResult = myResult.sort(['PassengerId'])
myResult.to_csv("H:/DATA/Python/Titanic/myResult.csv",index=False)