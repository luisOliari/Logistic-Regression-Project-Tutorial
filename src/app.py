from utils import db_connect
engine = db_connect()

# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', sep = ';')
df_copy=df.copy()

# Get basic info
df_copy.info()

# definir el marco general de los datos:
df_copy.head(10)

df_copy.dtypes

# chequeo de los datos estadisticos:
df_copy.describe()

#Revisando las categoricas 
print("Job:",df_copy.job.value_counts(),sep = '\n')
print("-"*40)
print("Marital:",df_copy.marital.value_counts(),sep = '\n')
print("-"*40)
print("Education:",df_copy.education.value_counts(),sep = '\n')
print("-"*40)
print("Default:",df_copy.default.value_counts(),sep = '\n')
print("-"*40)
print("Housing loan:",df_copy.housing.value_counts(),sep = '\n')
print("-"*40)
print("Personal loan:",df_copy.loan.value_counts(),sep = '\n')
print("-"*40)
print("Contact:",df_copy.contact.value_counts(),sep = '\n')
print("-"*40)
print("Month:",df_copy.month.value_counts(),sep = '\n')
print("-"*40)
print("Day:",df_copy.day_of_week.value_counts(),sep = '\n')
print("-"*40)
print("Previous outcome:",df_copy.poutcome.value_counts(),sep = '\n')
print("-"*40)
print("Outcome of this campaign:",df_copy.y.value_counts(),sep = '\n')
print("-"*40)

pip install missingno

import missingno as msno 
msno.matrix(df_copy)

# Parece que no tenemos ningún valor nulo excepto uno.
print('Data columns with null values:',df_copy.isnull().sum(), sep = '\n')

# visualizacion de la data:
# Configuring styles
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.facecolor'] = '#00000000'

plt.title("Distribution of Subscriptions")
sns.countplot(x="y", data=df_copy, palette="bwr")

fig = px.box(df_copy, x="job", y="duration", color="y")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

fig = px.scatter(df_copy, x="campaign", y="duration", color="y")
fig.show()

plt.bar(df_copy['month'], df_copy['campaign'])
plt.show()

sns.violinplot( y=df_copy["marital"], x=df_copy["cons.price.idx"] )

df_yes = df_copy[df_copy['y']=='yes']


df1 = pd.crosstab(index = df_yes["marital"],columns="count")    
df2 = pd.crosstab(index = df_yes["month"],columns="count")  
df3= pd.crosstab(index = df_yes["job"],columns="count") 
df4=pd.crosstab(index = df_yes["education"],columns="count")

fig, axes = plt.subplots(nrows=2, ncols=2)
df1.plot.bar(ax=axes[0,0])
df2.plot.bar(ax=axes[0,1])
df3.plot.bar(ax=axes[1,0])
df4.plot.bar(ax=axes[1,1]) 

f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df_copy.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)
plt.show()

plt.figure(figsize = (15, 30))
plt.style.use('seaborn-white')
ax=plt.subplot(521)
plt.boxplot(df_copy['age'])
ax.set_title('age')
ax=plt.subplot(522)
plt.boxplot(df_copy['duration'])
ax.set_title('duration')
ax=plt.subplot(523)
plt.boxplot(df_copy['campaign'])
ax.set_title('campaign')
ax=plt.subplot(524)
plt.boxplot(df_copy['pdays'])
ax.set_title('pdays')
ax=plt.subplot(525)
plt.boxplot(df_copy['previous'])
ax.set_title('previous')
ax=plt.subplot(526)
plt.boxplot(df_copy['emp.var.rate'])
ax.set_title('Employee variation rate')
ax=plt.subplot(527)
plt.boxplot(df_copy['cons.price.idx'])
ax.set_title('Consumer price index')
ax=plt.subplot(528)
plt.boxplot(df_copy['cons.conf.idx'])
ax.set_title('Consumer confidence index')
ax=plt.subplot(529)
plt.boxplot(df_copy['euribor3m'])
ax.set_title('euribor3m')
ax=plt.subplot(5,2,10)
plt.boxplot(df_copy['nr.employed'])
ax.set_title('No of employees')

numerical_features=['age','campaign','duration']
for cols in numerical_features:
    Q1 = df_copy[cols].quantile(0.25)
    Q3 = df_copy[cols].quantile(0.75)
    IQR = Q3 - Q1     

    filter = (df_copy[cols] >= Q1 - 1.5 * IQR) & (df_copy[cols] <= Q3 + 1.5 *IQR)
    df_copy=df_copy.loc[filter]

plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(df_copy['age'])
ax.set_title('age')
ax=plt.subplot(222)
plt.boxplot(df_copy['duration'])
ax.set_title('duration')
ax=plt.subplot(223)
plt.boxplot(df_copy['campaign'])
ax.set_title('campaign')

df_features=df_copy.copy()
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    df_features.loc[df_features['education'] == i, 'education'] = "middle.school"

df_features['education'].value_counts()

month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
df_features['month']= df_features['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
df_features['day_of_week']= df_features['day_of_week'].map(day_dict) 
df_features.loc[:, ['month', 'day_of_week']].head()

df_features.loc[df_features['pdays'] == 999, 'pdays'] = 0
df_features['pdays'].value_counts()

dictionary={'yes':1,'no':0,'unknown':-1}
df_features['housing']=df_features['housing'].map(dictionary)
df_features['default']=df_features['default'].map(dictionary)
df_features['loan']=df_features['loan'].map(dictionary)

dictionary1={'no':0,'yes':1}
df_features['y']=df_features['y'].map(dictionary1)

df_features.loc[:,['housing','default','loan','y']].head()

dummy_contact=pd.get_dummies(df_features['contact'], prefix='dummy',drop_first=True)
dummy_outcome=pd.get_dummies(df_features['poutcome'], prefix='dummy',drop_first=True)
df_features = pd.concat([df_features,dummy_contact,dummy_outcome],axis=1)
df_features.drop(['contact','poutcome'],axis=1, inplace=True)

df_features.loc[:,['dummy_telephone','dummy_nonexistent','dummy_success']].head()

df_job=df_features['job'].value_counts().to_dict()
df_ed=df_features['education'].value_counts().to_dict()

# Convirtió la frecuencia en pares de valores
df_features['job']=df_features['job'].map(df_job)
df_features['education']=df_features['education'].map(df_ed)

df_features.loc[:,['job','education']].head()

df_features.groupby(['marital'])['y'].mean()

ordinal_labels=df_features.groupby(['marital'])['y'].mean().sort_values().index
ordinal_labels

# Hemos ordenado las categorías en función de la media con respecto a nuestro resultado.
ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
ordinal_labels2

df_features['marital_ordinal']=df_features['marital'].map(ordinal_labels2)
df_features.drop(['marital'], axis=1,inplace=True)

df_features.marital_ordinal.value_counts()

df_scale=df_features.copy()
Categorical_variables=['job', 'education', 'default', 'housing', 'loan', 'month',
       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',
       'dummy_success', 'marital_ordinal']

feature_scale=[feature for feature in df_scale.columns if feature not in Categorical_variables]
scaler=StandardScaler()
scaler.fit(df_scale[feature_scale])

scaled_data = pd.concat([df_scale[['job', 'education', 'default', 'housing', 'loan', 'month',
       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',
       'dummy_success', 'marital_ordinal']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(df_scale[feature_scale]), columns=feature_scale)],
                    axis=1)
scaled_data.head()

X=scaled_data.drop(['y'],axis=1)
y=scaled_data.y

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score

# axis1=1 columnas =0 filas 
X=scaled_data.drop(['pdays','month','cons.price.idx','loan','housing','emp.var.rate','y'],axis=1)
y=scaled_data.y

# 2.1 Split the dataset so to avoid bias
X_train,X_test, y_train, y_test =train_test_split(X,y, random_state=1)
print("Input Training:",X_train.shape)
print("Input Test:",X_test.shape)
print("Output Training:",y_train.shape)
print("Output Test:",y_test.shape)

# 2.2 Join the train sets to ease insights
df_train = pd.concat((X_train, y_train), axis=1)

# 2.3 Get basic info
df_train.sample(10)

# 2.4 Describe the numerical and date variables
df_train.describe()

# 2.5 Perform univariate analysis - histograms
X_train.hist(figsize=(10,10), sharey=True)
plt.show()

from sklearn.linear_model import LogisticRegression

# Instantiate Logistic Regression
# tiene coeficientes y se puede explicar 
model = LogisticRegression(solver='liblinear')

X_train.info()

y_train

# Fit the data
# ajustar el modelo con X_train y y_train
model.fit(X_train, y_train)

# Make predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred
y_train_pred

# Check the accuracy score
# Accuracy test:
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score

accuracy_test=accuracy_score(y_test, y_test_pred)
accuracy_train=accuracy_score(y_train, y_train_pred)

print('Accuracy train:',accuracy_train)
print('Accuracy test:',accuracy_test)

# Confusion Matrix

banco_cm_train = confusion_matrix(y_train, y_train_pred,labels=model.classes_)
banco_cm_test = confusion_matrix(y_test, y_test_pred,labels=model.classes_)

banco_cm_train_norm=banco_cm_train / banco_cm_train.astype(np.float).sum(axis=1)
banco_cm_test_norm=banco_cm_test / banco_cm_test.astype(np.float).sum(axis=1)

banco_cm_train_norm
banco_cm_test_norm

from sklearn.metrics import ConfusionMatrixDisplay

np.set_printoptions(precision=2)

disp = ConfusionMatrixDisplay(confusion_matrix=banco_cm_train_norm,display_labels=model.classes_)
disp.plot()
plt.show()

np.set_printoptions(precision=2)

disp = ConfusionMatrixDisplay(confusion_matrix=banco_cm_test_norm,display_labels=model.classes_)
disp.plot()
plt.show()

