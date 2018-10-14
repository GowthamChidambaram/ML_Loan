#importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
loans=pd.read_csv("loan_data.csv")
print(loans.head())
print(loans.columns)
print(loans.describe())
print(loans.info())


#data analysis and visualization
plt.figure(figsize=(10,6))
loans[loans["credit.policy"]==1]["fico"].hist(bins=35,alpha=0.7)
loans[loans["credit.policy"]==0]["fico"].hist(bins=35,alpha=0.7)
plt.legend()
plt.show()
plt.figure(figsize=(10,6))
loans[loans["not.fully.paid"]==1]["fico"].hist(bins=35,alpha=0.7)
loans[loans["not.fully.paid"]==0]["fico"].hist(bins=35,alpha=0.7)
plt.legend()
plt.show()
plt.figure(figsize=(10,12))
sns.countplot(x="purpose",data=loans,hue="not.fully.paid")
plt.tight_layout()
plt.show()
sns.jointplot(x="fico",y="int.rate",data=loans)
plt.show()


#dealing with cat columns
cat_feaut=["purpose"]
train=pd.get_dummies(loans,columns=cat_feaut,drop_first=True)

print(train.head())
print(train.info())

#splitting training and testing data
x=train.drop("not.fully.paid",axis=1)
y=train["not.fully.paid"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#applying decission tree model
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
pred=tree.predict(x_test)
print("Result using random tree model:")
print("Classification report:")
print(classification_report(y_test,pred))
print("confusion matrix:")
print(confusion_matrix(y_test,pred))


#applying random forest
rf=RandomForestClassifier(n_estimators=300)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)
print("Result using random forest model:")
print("Classification report:")
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))



