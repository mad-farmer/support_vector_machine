import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("breast-cancer-wisconsin-data.csv")
#importing csv file
df.head()  
#show first 5 rows
print(df.columns) 
#gives column names in data
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#dropping nan and id column

y=df.diagnosis.values #classes
x=df.drop(["diagnosis"],axis=1) #features


#%% Visulation
m=df[df.diagnosis=="M"]
b=df[df.diagnosis=="B"]
# M=malignant, B=benign

plt.scatter(m.radius_mean,m.texture_mean,
            color="red",label="malignant",alpha=0.3)
plt.scatter(b.radius_mean,b.texture_mean,
            color="blue",label="benign",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()
# you can visualize other features


#%%
ax = sns.countplot(y,label="Count")       
#countplot


#%% correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(),annot=True,
            linewidths=.5,fmt='.1f',ax=ax)


#%%
df.diagnosis=[1 if each=="M" else 0 
              for each in df.diagnosis]
#M=1 and B=0


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.20,random_state=42)
#train and test split

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)
# Scaling features


#%% Classification

#MODEL WITH SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear') #linear
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy:",svclassifier.score(X_test,y_test))



#%%
svclassifier2 = SVC(kernel='poly',degree=3) #poly
svclassifier2.fit(X_train, y_train)
y_pred2 = svclassifier2.predict(X_test)

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
print("Accuracy:",svclassifier2.score(X_test,y_test))


#%%
svclassifier3 = SVC(kernel='rbf') #rbf
svclassifier3.fit(X_train, y_train)
y_pred3 = svclassifier3.predict(X_test)

print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
print("Accuracy:",svclassifier3.score(X_test,y_test))


#%%
svclassifier4 = SVC(kernel='sigmoid') #sigmoid
svclassifier4.fit(X_train, y_train)
y_pred4 = svclassifier4.predict(X_test)

print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))
print("Accuracy:",svclassifier4.score(X_test,y_test))

#%%



