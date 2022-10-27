-AI- FINAL - Ka - Scene
Quiz - 02:
df.fillna(df.mean())
#Quiz # 2
import numpy as np
import pandas as pd

car_data = pd.read_csv('Peugeot207.csv')

df = pd.DataFrame(car_data)

#First 5 Rows of Dataset
df.head(5)

# Check Null Values

df.isnull()

# Use unique Function
df.nunique()

# print last 5 data set values
df.tail()

# handle Categorical values if needed
data_num = df[['code','km','model','year','price']]
data_nonNum = df[['brand','color','gearbox','option']]

# Creating Dummy
dummy = pd.get_dummies(data_nonNum ,columns=['brand','color','gearbox','option'])
dummy

# Concat
CarNewData = pd.concat([data_num.iloc[:,1:],dummy],axis=1)
CarNewData

# train and test you data by using standard train_test_split()

from sklearn.model_selection import train_test_split
# independent


xn = [dummy.gearbox_Automatic]
xn


xxn = [dummy.gearbox_Manual]
type(xxn)

x = np.array(xn)
type(x)
# Dependent
yn = [df.price]
type(yn)

y = np.array(yn)
type(y)

x = final.drop(['Price'],axis=1)

y# train and test you data by using standard train_test_split()

from sklearn.model_selection import train_test_split
# independent
xn = [dummy.gearbox_Automatic]

xxn = [dummy.gearbox_Manual]
type(xxn)
x = np.array(xn)
type(x)
# Dependent
yn = [df.price]
type(yn)

y = np.array(yn)
type(y)

x = final.drop(['Price'],axis=1)

# Classification Algorithm for predicton
from sklearn.linear_model import LinearRegression

LinReg  = LinearRegression()
LinReg.fit(x,y)

y_pred= LinReg.predict(x)
y_pred

---------------------
Download any DataSet from Kaggle and apply Univariate Linear Regression.
Without Using Sklearn Library
FA19-BSCS-0046 Usama Saleem AI-Lab-AM

from google.colab import drive
drive.mount('/content/gdrive')
!ls

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/gdrive/MyDrive/Kaggle'

%cd /content/gdrive/My Drive/Kaggle
!ls

!kaggle datasets download -d prakharrathi25/home-prices-dataset

df = pd.read_csv('home_data.csv')
print('\n Number of row and Columns in the Data Set: ', df.shape)

xn = [df.sqft_living]
xn
type(xn)

x = np.array(xn)
type(x)



# y = df['price']
yn = [df.price]
type(yn)

y = np.array(yn)
type(y)

# number of observations/points
n = np.size(x)

# mean of x and y vector
m_x = np.mean(x)
m_y = np.mean(y)


# calculating SS_xy & SS_xx
SS_xy = (np.sum(y*x) - (n*(m_y)*(m_x)))
SS_xx = np.sum(x*x) - n*m_x*m_x 

# calculating regression coefficients

b = SS_xy / SS_xx
a = m_y - b*m_x

print("Value of a: ", a)
print("Value of b: ", b)


# Prediction Equation

y = b*x + a
plt.scatter(x,y)
#from sklearn.linear_model import LinearRegression

#LinReg  = LinearRegression()
#LinReg.fit(x,y)
#y_pred= LinReg.predict(x)
#y_pred
#plt.scatter(x,y)
#plt.plot(x,y_pred, color='black')

--------------------------------
Download any DataSet from Kaggle and apply Univariate Linear Regression.

By Using Sklearn Library
FA19-BSCS-0046 Usama Saleem
x = [df.sqft_living]
x
# y = df['price']
y = [df.price]
y
from sklearn.linear_model import LinearRegression
LinReg  = LinearRegression()
LinReg.fit(x,y)
y_pred= LinReg.predict(x)
y_pred
plt.scatter(x,y)
plt.plot(x,y_pred, color='black')
---------------------------------

MultiVarient Regression

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

df = pd.read_csv('homeprices.csv')
find_median = math.floor(df.bedrooms.median())
find_median
df.bedrooms = df.bedrooms.fillna(find_median)
df.shape

N = len(x)
x_mean = x.mean()
y_mean = y.mean()
    
B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den
    
B0 = y_mean - (B1*x_mean)

N = len(x)
num = (N * (x*y).sum()) - (x.sum() * y.sum())
den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
R = num / den

print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)

y = B0 + B1 * new_x
    
reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
--------------------------------------------



------------With Sklearn --------------------------
my_model = LinearRegression()
x = df[['area','bedrooms','age']]
y = df.price
my_model.fit(x,y)
y_pred = my_model.predict(x)
print(y_pred)
my_model.score(x,y)

x1 = df['area']
x2 = df['bedrooms']
x3 = df['age']


plt.scatter(df.area,y)
plt.plot(x1,y_pred,color='black')
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


plt.scatter(df.bedrooms,y)
plt.plot(x2,y_pred,color='black')
plt.title('Bedrooms vs Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()

plt.scatter(df.area,y)
plt.plot(x3,y_pred,color='black')
plt.title('Age vs Price')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()

--------------------------------------
4. Find the number of null values in each column 
df.isnull().sum()

5. Find the important information about data value types
print(df.info())

6. Unique
df.nunique()

7. Create Two separate data frames

-----------------------------------
Lab 9-----
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data_set.csv')
df.head()


df.isnull().sum()
sns.heatmap(df.isnull())
df.info()
df.nunique()
df.head()
df.describe()
data_numerical = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
data_nonNumerical = df[['Loan_ID','Gender','Married', 'Dependents', 'Education', 'Self_Employed','Property_Area', 'Loan_Status' ]]
df.head()
data_numerical.columns

data_numerical.head(5)
data_nonNumerical.head(5)

data_nonNumerical.groupby('Gender')['Loan_ID'].count()

# 13. Set the null values in Gender column with the majority value 
For categorical column 
x = data_nonNumerical.columns
for i in x[1:]:
    m=data_nonNumerical.groupby(i)['Loan_ID'].count().idxmax()
    data_nonNumerical[i].fillna(m,inplace=True)


data_nonNumerical.isnull().sum()

14. To calculate the mean for numeric columns 
 data_numerical[Column name].mean()

15. Now, nan values of numeric columns fill with mean of columns
x = data_numerical.columns
for i in x:
    m=data_numerical[i].mean()
    data_numerical[i].fillna(m,inplace=True)


data_numerical.isnull().sum()

dummies = pd.get_dummies(data_nonNumerical ,columns=['Gender','Married', 'Dependents', 'Education', 'Self_Employed','Property_Area', 'Loan_Status'])

dummies.drop(['Loan_ID'],axis=1)

sns.heatmap(data_numerical.isnull())

sns.heatmap(data_nonNumerical.isnull())

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
data_numerical.iloc[:,1:] = sc.fit_transform(data_numerical.iloc[:,1:])

df_loan = pd.concat([data_numerical.iloc[:,1:],dummies],axis=1)
df_loan.head()

from sklearn.model_selection import train_test_split
x=df_loan.drop(['Loan_Status_N','Loan_Status_Y'],axis=1)
y=df_loan['Loan_Status_Y']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=30)   #0.2  = 20%

---------------------------
Import the model and create the object:
form sklearn.svm import SVC
svc_model=SVC()

- Train and test the model
from sklearn.metrics import classification_report,confusion_matric, accuracy_score

print(confusion_matrix(y_test,svm_pred)
print(classification_report(y_test,svm_pred))
print(accuracy_score(y_test, svm_pred))

- Create KNN, Navie Bayes

Navie = navie_bayes.GaussianNB()

knn = kNeighboursClassifier()n)neighbours=3)
dtc = DesisionTreeClassifier()



Decision Tree:
# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTreeClassifier(random_state =0)
# tree = dtree.fit(X_train,y_train)

# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train,y_train)

df_loan.info()
-----------------------------------------------

--------------------------------------------------------------------
Question 2:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

std = pd.DataFrame({"StdID":[1001,1001,1002],
"FName":["Ali","Ali","Ahmed"],
"LName":["Syed","Syed","Alam"],
"Course":["DBMS","Artificial Intelligence","DBMS"],
"Grade":["B+","A","C-"],
"RegDate":["7-2-21","9-2-21","6-2-21"],
"Semester":["SP21","SP21","SP21"],
"Attendence":["70%","80%","65%"]})

std

std["Name"] = std["FName"]+" "+std["LName"]

grades = ['A','A-','B+','B','B-','C+','C','C-','D+','D','F']

from pandas.api.types import CategoricalDtype
dtype = CategoricalDtype(categories=grades, ordered=True)
std.Grade = std.Grade.astype(dtype).cat.codes

std.Attendence = pd.to_numeric(std.Attendence.str.replace('%',''))

std.groupby('Course')['Attendence'].mean()

std.groupby('Course')['Attendence'].agg([np.mean])

--------------------------------------
----------------
Quiz # 03
#import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Q1. Read data from Q4_data.csv into the dataframe. (2 points)
df = pd.read_csv('Q4_data.csv')
df.head()

Q2. Display the description of numerical columns. (2 points)
df.describe()

Q3. Display information of all the columns. (1 point)
df.info()

Q4. Perform Correlation Analysis between all pairs of variables. Identify (by printing) what two input variables are highly
correlated to the output variable Final. (4 points)
df.corr()

Q5. Split the dataset into input and output variables. Then input the data into training and testing data set. Use 70% of the
data as training data and rest as testing data. Use 42 as random seed. (5 points)

x = df[['Quiz-1','Quiz-2','Quiz-3','Quiz-4','Major-1','Major-2','Major-3']]
y = df.Final

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

my_model = LinearRegression()

my_model.fit(x,y)

y_pred = my_model.predict(x)
print('Predicted Value:', y_pred)

my_model.score(x,y)

Q6. Using the scikit library function, apply Linear Regression between Major-1 and Final. Use all the data, print the
slope and the intercept. (Hint: Not to use the data from Q5). (6 points)


x = df['Major-1']
y = [df.Final]
from sklearn.linear_model import LinearRegression
LinReg  = LinearRegression()
LinReg.fit(x,y)
y_pred= LinReg.predict(x)
y_pred
---------------------------------------------


--------LR----------

Linear Regression:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_csv
df.head()
df.plot.scatter(x='MID',y='FINAL')
df.corr()
x = df.MID
y = df.OVERALL
y = y.dropna(axis=0)

-------------- Code -----
def compute_coff(x,y):
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    SS_yy = np.sum(y*y) - n*m_y*m_y

    # calculating regression coefficients
    b = SS_xy / SS_xx
    a = m_y - b*m_x
    
    #calculating r and r_sq
    r = SS_xy/math.sqrt(SS_xx*SS_yy)
    r_sq = b*SS_xy/SS_yy
    
    return a,b,r,r_sq

a,b,r,r_squared = compute_coff(x,y)
print("Slope:",b)
print("Intercept:",a)
print("R:",r)
print("R Squared:",r_squared)


---------------------------Code---
---------Library----

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
marks = pd.read_csv('marks.csv')
marks.describe()

#separating dependent and independent variables
X = marks.MID.dropna()
y = marks.OVERALL.dropna()
#Data Preparation
X = X.values.reshape(-1,1)
y= y.values.reshape(-1,1)

# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x_train, y_train)
# Predict
y_predicted = regression_model.predict(x_test)

# model evaluation
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Mean squared error: ', mse)
print('Root mean squared error: ', mse**(1/2.0))
print('R2 score: ', r2)

-------Linear Regression-------

--------Logit and Clustering----------
import pandas as pd
import numpy as np
data = pd.read_csv("StudentGrades.csv")

data.columns

X = data[['age', 'sex', 'famsize', 'Pstatus', 'traveltime',
       'studytime', 'failures', 'internet', 'Dalc', 'Walc', 'health',
       'absences', 'G1', 'G2', 'G3']]
y = np.where(data.Total>=40,1,0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_Train, X_Test,y_train ,y_test = train_test_split(X,y,test_size=0.25)


clf = LogisticRegression(max_iter=500,random_state=0).fit(X_Train, y_train)
#clf.score(X_Test, y_test)

#clf.predict_proba(X[4:5])
y_pred = clf.predict(X_Test)

--------Model Evaluation--------
from sklearn.metrics import confusion_matrix
#y_true = [2, 0, 2, 2, 0, 1]
#y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_test, y_pred)

--------Clustering--------

x = [2,2,8,5,7,6,1,4]
y = [10,5,4,8,5,4,2,9]
import matplotlib.pyplot as plt
plt.scatter(x,y)

def distance(x1,y1,x2,y2):
  return ((x2-x1)**2+(y2-y1)**2)**(1/2.0)

distance(2,5,1.5,3.5)

---------Using Library-----------
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[2, 10], [2, 5], [8, 4],
               [5, 8], [7, 5], [6, 4],[1,2],[4,9]])

kmeans = KMeans(n_clusters=3, max_iter=1, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[6, 6], [3, 3]]))
print(kmeans.cluster_centers_)


---- Silhouette Coefficient ----


from sklearn.metrics import silhouette_score
score = silhouette_score(X, kmeans.labels_, metric='euclidean')

print('Silhouetter Score: %.3f' % score)

-------------------------------------------

------CLustering-------------------
import numpy as np
import pandas as pd

car_data = pd.read_csv('Peugeot207.csv')

df = pd.DataFrame(car_data)

#First 5 Rows of Dataset
df.head(5)

# Check Null Values

df.isnull().any()

# Use unique Function
df.nunique()

# print last 5 data set values
df.tail()

plt.rcParams['figure.figsize'] = (14,7)
sns.set(style = 'whitegrid')
sns.displot(df['Annual income (K$)'])

plt.show()

Show the percentage of male and female visitors using the pie chart
labels = ['F','M']
size = df['Gender'].value_counts()
colors = ['lightgreen','orange']
explode = [0,0.1]


plt.rcParams['figure.fig.size'] = (0,6)
plt.pie(size, colors = colors, explode, labels = labels, shadow=true,  autopct = '%.2%%')
plt.title('Gender',fontsize = 20 )
plt.legend()
plt.show()

countplot to show the more clear distribution of age and annual income


plt.rcParams['figure.fig.size'] = (15,8)
sns.countplot(df['Age'], palette = 'hsv')
plt.title('Distribution of Age', fontsize = 20)
plt.show()

Pairplot is used to understand the relationship between different factors of the dataset. Show the plot using plt.show method.
sns.pariplot(df)

Show the heatmap to represent the correlation between the features of the dataset. To pass the data use .corr() method
sns.heatmap(df.corr(),cmap = 'Wistia', annot=True)

Compare the spending scores of male and females with boxen plot.

plt.rcParams['figure.figsize'] = (18,7)
sns.boxenplot(df['Gender'],df['Spending score (1-100)'], palette = 'Blues')


Compare the annual income of male and female customers visiting the mall using the violinplot.
plt.rcParams['figure.figsize'] = (18,7)
sns.violinplot(df['Gender'],df['Spending score (1-100)'], palette = 'rainbow')

Use the line plot to show the relationship between annual income vs age and annual income vs spending score
x = df['Annual income (k$)']
y = df['Age']
z = df['Spending sccore (1 - 100)']

sns.lineplot(x,y, color='blue')
sns.lineplot(x,z, color='pink')

plt.show()


Above visualizations will help a BI analyst to understand the relationship and dependencies between features. Now let’s create a K-means clustering model. But to select the value of K, use the elbow technique.
x = df.iloc[:,[3,4]].values

The code below will import KMeans and create the clusters with different values of K from 1 to 11 and append the variance in an array.
from.sklearn.cluster import KMeans

wcss  = []

for i in range(1,11)
    km = KMeans(n_clusters=i)
    km.fit(x)
    wcss.append(km.inertia_)


By plotting it on graph, we’ll get an optimal value of K

plt.plot(range(1,11),wcss)
plt.title('Method', fontsize = 20)
plt.xlabel('No. of clusters')

plt.show()


If we select the K=5 from above graph. Create a KMeans model with five cluster.
km = KMeans(n_clusters = 5)
y_means = km.fit_predict(x)


Let’s show the clusters on a graph.

plt.scatter(x[y_means == 0, 0], x[y_means == 0,1], s =100, c='pink')
plt.scatter(x[y_means == 1, 0], x[y_means == 1,1], s =100, c='yellow')
plt.scatter(x[y_means == 2, 0], x[y_means == 2,1], s =100, c='cyan')
plt.scatter(x[y_means == 3, 0], x[y_means == 3,1], s =100, c='magenta')
plt.scatter(x[y_means == 4, 0], x[y_means == 4,1], s =100, c='orange')
plt.scatter(km.cluster_center_[:,0], km.cluster_center_[:,1], s =50, c='blue',label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means CLustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('spending score')
plt.show()



****************************************************************************
1- ML Classification:
****************************************************************************





****************************************************************************
2- Data Visualization:
****************************************************************************
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('wheat.data')
df.wheat_type.unique()
df.wheat_type.value_counts()
df1 = df[['area','perimeter']]
df2 = df[['groove','asymmetry']]
df.plot.scatter()
df1.area.count()

1-histogram
df.perimeter.plot.hist()
df[['area','perimeter']].plot.hist(alpha=0.80)

2-Scatter Plot
df.plot.scatter('length','width')
df[['area','perimeter']].plot.scatter('perimeter','area')

3-3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry,c='r')
fig.show()

4-Andrew Plot
pd.plotting.andrews_curves(df,'wheat_type')

5-Box plot
df.columns
f['PetalWidth'].plot.hist()
df.boxplot(column=[ 'length', 'width',
       'asymmetry', 'groove'])
box3 = f.boxplot(column=['SepalLength','SepalWidth','PetalLength','PetalWidth'])

6-Bar Graph
df4.health.value_counts().plot(kind='bar')
df4.age.plot.hist()

7-Line Graph
import pandas as pd
import matplotlib.pyplot as plt
   
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
        'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
       }
  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
  
plt.plot(df['Year'], df['Unemployment_Rate'], color='green', marker='o')
plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
#plt.show()

8-Pie PLot
df['wheat_type'].value_counts().plot.pie()



****************************************************************************
3- Neural Network:
****************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/gdrive/MyDrive/Kaggle'

%cd /content/gdrive/My Drive/Kaggle
!ls

df=pd.read_csv("Iris.csv")

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models

x = df.iloc[:,0:4].values
y = df.iloc[:,5].values

print(x.shape)
print(y.shape)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
y1

y = pd.get_dummies(y1).values
print(y[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'relu'),
    tf.keras.layers.Dense(3,activation = 'softmax')
])



model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 50, epochs = 100)
loss,accuracy = model.evalute(X_test)










****************************************************************************

****************************************************************************
4- Searching Algorithm:
****************************************************************************
1- Hill climbing:
****************************************************************************
# Hill Climbing Ai lab
# Usama Saleem 
# FA19 - BSCS - 0046
# Python program to implement Hill Climbing algorithm for Travel Salesman Problem.

# Funtion 1
import random

def random_Select(TravelValues):
    cities = list(range(len(TravelValues)))
    solutionValues = []
    
    for i in range(len(TravelValues)):
        randomCity = cities[random.randint(0,len(cities) - 1)]
        solutionValues.append(randomCity)
        cities.remove(randomCity)
    return solutionValues
    
    
# Funtion 2
def  route_Lenght(TravelValues, solutionValues):
    routelenth = 0
   # lenth = len(TravelValues)
    for i in range(len(solutionValues)):
         lenth += TravelValues[solution[i-1]][solution[i]]
    return lenth

# Funtion 3
def neighbour(solutionValues):
    neighbours = []
    for i in range(len(solutionValues)):
        for j in range(i+1, len(solutionValues)):
            neighbour=solutionValues.copy()
            neighbour[i]= solutionValues[j]
            neighbour[j]= solutionValues[i]
            neighbours.append(neighbour)
    return neighbours
 


 # Funtion 4

def best_neighbour(TravelValues, neighbours):
    bestRouteLength = routeLength(tsp, neighbours[0])
    bestNeighbour = neighbours[0]
    
    for neighbour in neighbours:
        currentRouteLength = routeLength(tsp, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
            
    return bestNeighbour, bestRouteLength
    
    
def hillClimbing(TravelValues):
    currentSolution = random_Select(TravelValues)
    currentRouteLength = route_Lenght(TravelValues, solutionValues)
    neighbours = neighbour(solutionValues)
    bestNeighbour, bestNeighbourRouteLength = best_neighbour(TravelValues, neighbours)

    while bestNeighbourRouteLength < currentRouteLength:
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours)

    return currentSolution, currentRouteLength
    



def main():
    
    TravelValues = [
        [0,400,500,300],
        [400,0,300,500],
        [500,300,0,400],
        [300,500,400,0]
    ]
    
    print(hillClimbing(TravelValues))

if __name__ == "__main__":
     main()
    
----------------------------
 Question 2.
# 2. Explain Hill climber code above have some limitation?

# Answer: 
#  The above code of hill climb has some limitations and often gets stuck
# local maximum, and flat state-space region as there is no uphill to go, algorithm often gets lost in the plateau.




****************************************************************************

****************************************************************************
2- A*
****************************************************************************
# Write Python program to implement Greedy Best First Search Algorithms for following rout
# finding problem from Arad to Bucharest.

from queue import PriorityQueue

mapPoints = {'Arad':{'Zerind':75,'Timisoara':118,'Sibiu':140},
         'Zerind':{'Oradea':71,'Arad':75},
         'Eforie':{'Hirsova':86}
}

def aStar(source, destination):
         straight_line ={
                        'Arad': 366,
               		 'Eforie':161
                             }

         priQue, visited = PriorityQueue(), {}
         priQue.put((straight_line[source], 0, source, [source]))

         visited[source] = straight_line[source]
         while not priQue.empty():
             (heuristic, cost, vertex, path) = priQue.get()
             print('Queue Status:',heuristic, cost, vertex, path)
             if vertex == destination:
                 return heuristic, cost, path
             for next_node in mapPoints[vertex].keys():
                 current_cost = cost + mapPoints[vertex][next_node]
                 heuristic = current_cost + straight_line[next_node]
                 if not next_node in visited or visited[
                     next_node] >= heuristic:
                     visited[next_node] = heuristic
                     priQue.put((heuristic, current_cost, next_node, path + [next_node]))

def main():
    print('Source :', end=' ')
    source = input().strip()
    print('Destination :', end=' ')
    goal = input().strip()
    if source not in mapPoints or goal not in mapPoints:
        print('NOT EXIST.')
    else:
        heuristic, cost, optimal_path = aStar(source, goal)
        print('Min of total heuristic_value =', heuristic)
        print('Total min cost =', cost)
        print('\nRoute:')
        print(optimal_path)

main()








****************************************************************************

****************************************************************************
3-Greedy Best First Search
****************************************************************************
- ----------
def coin_change_greedy(n):
  coins = [20, 10, 5, 1]
  i = 0

  while(n>0):
    if(coins[i] > n):
      i = i+1
    else:
      print(coins[i])
      n = n-coins[i];
  print("\n\n\n\n")

if __name__ == '__main__':
  for i in range(1, 21):
    coin_change_greedy(i)


-------

# Write Python program to implement Greedy Best First Search Algorithms for following rout
# finding problem from Arad to Bucharest.

mapPoints = { 'Arad' :[['Zerind',374],['Timisoara',329],['Sibiu',253]],
         'Zerind':[['Oradea',380],['Arad',366]],
         'Eforie':[['Hirsova',151]]
}

def greedyBestFirstSearch(mapPoints, start, end):
    explored = [] # Path store karta hai
    queue = [start]
    while queue:
        print (queue)
        node = queue.pop(0) 
        if node not in explored:
            explored.append(node)
            if node == end:
                break
            neighbors = mapPoints[node]
            neighbors.sort(key=lambda a: a[1])
            print (neighbors)
            queue=neighbors.pop(0)

    print(explored)

Start = input('Enter Starting node  : ')
End = input('Enter Ending node  : ')
print()
print('Greedy Best First Search from staring node to goal node')
greedyBestFirstSearch(mapPoints, Start, End)





****************************************************************************






Remaining:


# Write Python program to implement Greedy Best First Search Algorithms for following rout
# finding problem from Arad to Bucharest.

from queue import PriorityQueue

mapPoints = {'Arad':{'Zerind':75,'Timisoara':118,'Sibiu':140},
         'Zerind':{'Oradea':71,'Arad':75},
         'Oradea':{'Sibiu',151},
         'Sibiu':{'Rimniciu Vilcea':80,'Fagaras':99,'Arad':140},
         'Fagaras':{'Sibiu':99,'Bucharest':211},
         'Rimniciu Vilcea':{'Pitesti':97,'Craiova':146,'Sibiu':80},
         'Timisoara':{'Lugoj':111,'Arad':118},
         'Lugoj':{'Mehadia':70},
         'Mehadia':{'Lugoj':70,'Dorbeta':75},
         'Dobreta':{'Mehadia':75,'Craiova':120},
         'Pitesti':{'Craiova':138,'Bucharest':101},
         'Craiova':{'Pitesti':138,'Dobreta':120,'Rimniciu Vilcea':146},
         'Bucharest':{'Giurgiu':90,'Urziceni':85,'Fagaras':211,'Pitesti':101},
         'Giurgiu': {'Bucharest':90},
         'Urziceni':{'Vaslui':142,'Hirsova':98,'Bucharest':85},
         'Vaslui':{'Lasi':92,'Urziceni':142},
         'Lasi':{'Neamt':87,'Vaslui':92},
         'Neamt':{'Lasi':87},
         'Hirsova':{'Eforie':86,'Urziceni':98},
         'Eforie':{'Hirsova':86}
}

def aStar(source, destination):
         straight_line ={
                        'Arad': 366,
                        'Zerind': 374,
                        'Oradea': 380,
                        'Sibiu':  253,
                        'Fagaras':176,
                        'Rimniciu Vilcea': 193,
                        'Timisoara': 329,
                        'Lugoj': 244,
                        'Mehadia': 241,
                        'Dobreta': 242,
                        'Pitesti':100,
                        'Craiova':160,
                        'Bucharest':0,
                        'Giurgiu':77,
                        'Urziceni': 80,
                        'Vaslui':199,
                        'Lasi':226,
                        'Neamt':234,
                        'Hirsova':151,
                        'Eforie':161
                             }

         priQue, visited = PriorityQueue(), {}
         priQue.put((straight_line[source], 0, source, [source]))

         visited[source] = straight_line[source]
         while not priQue.empty():
             (heuristic, cost, vertex, path) = priQue.get()
             print('Queue Status:',heuristic, cost, vertex, path)
             if vertex == destination:
                 return heuristic, cost, path
             for next_node in mapPoints[vertex].keys():
                 current_cost = cost + mapPoints[vertex][next_node]
                 heuristic = current_cost + straight_line[next_node]
                 if not next_node in visited or visited[
                     next_node] >= heuristic:
                     visited[next_node] = heuristic
                     priQue.put((heuristic, current_cost, next_node, path + [next_node]))

def main():
    print('Source :', end=' ')
    source = input().strip()
    print('Destination :', end=' ')
    goal = input().strip()
    if source not in mapPoints or goal not in mapPoints:
        print(' CITY DOES NOT EXIST.')
    else:
        heuristic, cost, optimal_path = aStar(source, goal)
        print('Min of total heuristic_value =', heuristic)
        print('Total min cost =', cost)
        print('\nRoute:')
        print(optimal_path)

main()







-------------------------------------------------









# Write Python program to implement Greedy Best First Search Algorithms for following rout
# finding problem from Arad to Bucharest.

mapPoints = { 'Arad' :[['Zerind',374],['Timisoara',329],['Sibiu',253]],
         'Zerind':[['Oradea',380],['Arad',366]],
         'Oradea':[['Sibiu',253]],
         'Sibiu':[['Rimniciu Vilcea',193],['Fagaras',178],['Arad',366]],
         'Fagaras':[['Sibiu',253],['Bucharest',0]],
         'Rimniciu Vilcea':[['Pitesti',98],['Craiova',160],['Sibiu',253]],
         'Timisoara':[['Lugoj',244],['Arad',366]],
         'Lugoj':[['Mehadia',241]],
         'Mehadia':[['Lugoj',244],['Dorbeta',242]],
         'Dobreta':[['Mehadia',241],['Craiova',160]],
         'Pitesti':[['Craiova',160],['Bucharest',0]],
         'Craiova':[['Pitesti',98],['Dobreta',242],['Rimniciu Vilcea',193]],
         'Bucharest':[['Giurgiu',77],['Urziceni',80],['Fagaras',178],['Pitesti',98]],
         'Giurgiu': [['Bucharest',0]],
         'Urziceni':[['Vaslui',199],['Hirsova',151],['Bucharest',0]],
         'Vaslui':[['Lasi',226],['Urziceni',80]],
         'Lasi':[['Neamt',234],['Vaslui',199]],
         'Neamt':[['Lasi',226]],
         'Hirsova':[['Eforie',161],['Urziceni',80]],
         'Eforie':[['Hirsova',151]]
}

def greedyBestFirstSearch(mapPoints, start, end):
    explored = [] # Path store karta hai
    queue = [start]
    while queue:
        print (queue)
        node = queue.pop(0) 
        if node not in explored:
            explored.append(node)
            if node == end:
                break
            neighbors = mapPoints[node]
            neighbors.sort(key=lambda a: a[1])
            print (neighbors)
            queue=neighbors.pop(0)

    print(explored)

Start = input('Enter Starting node  : ')
End = input('Enter Ending node  : ')
print()
print('Greedy Best First Search from staring node to goal node')
greedyBestFirstSearch(mapPoints, Start, End)


-------------------------------------------
A*


import numpy as np

class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position

#This function return the path of the search
def return_path(current_node,maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0
    # we update the pn(path)):
        #pathof start to end found by A-star search with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1
    return result


def search(maze, cost, start, end):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """

    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration. 
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []  
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = [] 
    
    # Add the start node
    yet_to_visit_list.append(start_node)
    
    # Adding a stop condition. This is to avoid any infinite loop and stop 
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . serarch movement is left-right-top-bottom 
    #(4 movements) from every positon

    move  =  [[-1, 0 ], # go up
              [ 0, -1], # go left
              [ 1, 0 ], # go down
              [ 0, 1 ]] # go right


    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent
            
            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    #find maze has got how many rows and columns 
    no_rows, no_columns = np.shape(maze)
    
    # Loop until you find the end
    
    while len(yet_to_visit_list) > 0:
        
        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1    

        
        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
                
        # if we hit this point return the path such as it may be no solution or 
        # computation cost is too high
        if outer_iterations > max_iterations:
            print ("giving up on pathfinding too many iterations")
            return return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node,maze)


        # Generate children from all adjacent squares
        children = []

        for new_position in move: 

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or 
                node_position[0] < 0 or 
                node_position[1] > (no_columns -1) or 
                node_position[1] < 0):
                continue
                """
                row 
                 5-1 =4 0---->4  >4 
                 0--->4 <0 
                 column 6-1 =5  0 - 5 >5
                 0-5 <0                   """

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            
            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost
            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) + 
                       ((child.position[1] - end_node.position[1]) ** 2)) 

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)


if __name__ == '__main__':

    maze = [[0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]]

            
    
    start = [0, 0] # starting position
    end = [4,5] # ending position
    cost = 1 # cost per movement

    path = search(maze,cost, start, end)
    print(path)






----------------------------------------
A Search Short

class box():
    """A box class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given board"""

    # Create start and end node
    start_node = box(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = box(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = box(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():

    board = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (6, 6)

    path = astar(board, start, end)
    print(path)


if __name__ == '__main__':
    main()

------------------------------------------------------


-------Complete IDS File---------
------------------------AI------------------------------------------

Linear Regression:

from google.colab import drive
drive.mount('/content/gdrive')
!ls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/gdrive/MyDrive/Kaggle'

%cd /content/gdrive/My Drive/Kaggle
!ls
!kaggle datasets download -d prakharrathi25/home-prices-dataset
df = pd.read_csv('home_data.csv')
print('\n Number of row and Columns in the Data Set: ', df.shape)
df.head(10)
Same in both**********
x = [df.sqft_living]
x
# y = df['price']
y = [df.price]
y
from sklearn.linear_model import LinearRegression
LinReg  = LinearRegression()
LinReg.fit(x,y)
y_pred= LinReg.predict(x)
y_pred
plt.scatter(x,y)
plt.plot(x,y_pred, color='black')
*****************************************************
Using Sklearn Library
xn = [df.sqft_living]
xn
type(xn)

x = np.array(xn)
type(x)

# y = df['price']
yn = [df.price]
type(yn)

y = np.array(yn)
type(y)

# number of observations/points
n = np.size(x)

# mean of x and y vector
m_x = np.mean(x)
m_y = np.mean(y)

# calculating SS_xy & SS_xx
SS_xy = (np.sum(y*x) - (n*(m_y)*(m_x)))
SS_xx = np.sum(x*x) - n*m_x*m_x 


# calculating regression coefficients

b = SS_xy / SS_xx
a = m_y - b*m_x

print("Value of a: ", a)
print("Value of b: ", b)

# Prediction Equation

y = b*x + a

plt.scatter(x,y)

***********************Multivarient linear regression*************************
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

df = pd.read_csv('homeprices.csv')
df
find_median = math.floor(df.bedrooms.median())
find_median
df.bedrooms = df.bedrooms.fillna(find_median)
df.shape
df
*/*/*-*-*-/-/*-/-*/*-/*/*-/-*/-Same in both===========**************----------/*----***-/

N = len(x)
x_mean = x.mean()
y_mean = y.mean()
    
B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den
    
B0 = y_mean - (B1*x_mean)

N = len(x)
num = (N * (x*y).sum()) - (x.sum() * y.sum())
den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
R = num / den

print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)

y = B0 + B1 * new_x
    
reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))


N = len(x)
x_mean = x.mean()
y_mean = y.mean()

B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den

x = df[['area','bedrooms','age']]
y = df.price


x1 = df['area']
x2 = df['bedrooms']
x3 = df['age']


plt.scatter(df.area,y)
plt.plot(x1,y_pred,color='black')
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

plt.scatter(df.bedrooms,y)
plt.plot(x2,y_pred,color='black')
plt.title('Bedrooms vs Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()

plt.scatter(df.area,y)
plt.plot(x3,y_pred,color='black')
plt.title('Age vs Price')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()


------------With Sklearn --------------------------
my_model = LinearRegression()
x = df[['area','bedrooms','age']]
y = df.price
my_model.fit(x,y)
y_pred = my_model.predict(x)
print(y_pred)
my_model.score(x,y)

x1 = df['area']
x2 = df['bedrooms']
x3 = df['age']


plt.scatter(df.area,y)
plt.plot(x1,y_pred,color='black')
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


plt.scatter(df.bedrooms,y)
plt.plot(x2,y_pred,color='black')
plt.title('Bedrooms vs Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()

plt.scatter(df.area,y)
plt.plot(x3,y_pred,color='black')
plt.title('Age vs Price')
plt.xlabel('Age')
plt.ylabel('Price')
plt.show()

--------------------------------------
Lab 9-----
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data_set.csv')
df.head()


df.isnull().sum()
sns.heatmap(df.isnull())
df.info()
df.nunique()
df.head()
df.describe()
data_numerical = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
data_nonNumerical = df[['Loan_ID','Gender','Married', 'Dependents', 'Education', 'Self_Employed','Property_Area', 'Loan_Status' ]]
df.head()
data_numerical.columns

data_numerical.head(5)
data_nonNumerical.head(5)

data_nonNumerical.groupby('Gender')['Loan_ID'].count()


x = data_nonNumerical.columns
for i in x[1:]:
    m=data_nonNumerical.groupby(i)['Loan_ID'].count().idxmax()
    data_nonNumerical[i].fillna(m,inplace=True)


data_nonNumerical.isnull().sum()


x = data_numerical.columns
for i in x:
    m=data_numerical[i].mean()
    data_numerical[i].fillna(m,inplace=True)



data_numerical.isnull().sum()

dummies = pd.get_dummies(data_nonNumerical ,columns=['Gender','Married', 'Dependents', 'Education', 'Self_Employed','Property_Area', 'Loan_Status'])

dummies.drop(['Loan_ID'],axis=1)

sns.heatmap(data_numerical.isnull())

sns.heatmap(data_nonNumerical.isnull())

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
data_numerical.iloc[:,1:] = sc.fit_transform(data_numerical.iloc[:,1:])

df_loan = pd.concat([data_numerical.iloc[:,1:],dummies],axis=1)
df_loan.head()

from sklearn.model_selection import train_test_split
x=df_loan.drop(['Loan_Status_N','Loan_Status_Y'],axis=1)
y=df_loan['Loan_Status_Y']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=30)   #0.2  = 20%


Decision Tree:
# from sklearn.tree import DecisionTreeClassifier
# dtree = DecisionTreeClassifier(random_state =0)
# tree = dtree.fit(X_train,y_train)

# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train,y_train)

df_loan.info()
-------------------------AI CODE-------------------------------------------
Visualization

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('wheat.data')
df.wheat_type.unique()
df.wheat_type.value_counts()
df1 = df[['area','perimeter']]
df2 = df[['groove','asymmetry']]
df.plot.scatter()
df1.area.count()

-histogram
df.perimeter.plot.hist()
df[['area','perimeter']].plot.hist(alpha=0.80)

-Scatter Plot
df.plot.scatter('length','width')
df[['area','perimeter']].plot.scatter('perimeter','area')

-3D plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry,c='r')
fig.show()

-Andrew Plot
pd.plotting.andrews_curves(df,'wheat_type')

-Box plot
df.columns
f['PetalWidth'].plot.hist()
df.boxplot(column=[ 'length', 'width',
       'asymmetry', 'groove'])
box3 = f.boxplot(column=['SepalLength','SepalWidth','PetalLength','PetalWidth'])

-Bar Graph
df4.health.value_counts().plot(kind='bar')
df4.age.plot.hist()

-Line Graph
import pandas as pd
import matplotlib.pyplot as plt
   
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010],
        'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
       }
  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
  
plt.plot(df['Year'], df['Unemployment_Rate'], color='green', marker='o')
plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
#plt.show()

-Pie PLot
df['wheat_type'].value_counts().plot.pie()
--------------------------------------------------------------------
--------------------------------------------------------------------
Question 2:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

std = pd.DataFrame({"StdID":[1001,1001,1002],
"FName":["Ali","Ali","Ahmed"],
"LName":["Syed","Syed","Alam"],
"Course":["DBMS","Artificial Intelligence","DBMS"],
"Grade":["B+","A","C-"],
"RegDate":["7-2-21","9-2-21","6-2-21"],
"Semester":["SP21","SP21","SP21"],
"Attendence":["70%","80%","65%"]})

std

std["Name"] = std["FName"]+" "+std["LName"]

grades = ['A','A-','B+','B','B-','C+','C','C-','D+','D','F']

from pandas.api.types import CategoricalDtype
dtype = CategoricalDtype(categories=grades, ordered=True)
std.Grade = std.Grade.astype(dtype).cat.codes

std.Attendence = pd.to_numeric(std.Attendence.str.replace('%',''))

std.groupby('Course')['Attendence'].mean()

std.groupby('Course')['Attendence'].agg([np.mean])
------------------
Quiz # 02
Read Data Set
# Usama Saleem FA19-Bscs-0046
import numpy as np 
import pandas as pd 

df = pd.read_excel("New_Covid19.xlsx")
df
df.dtypes

df.dropna()
df.dropna(axis=1)

df
df.shape
Histogram breaking your age in 5 bin.
import matplotlib.pyplot as plt

plt.hist(df.age[(df['positive'].notnull())], 5, Facecolor='blue')
plt.xlabel('Age')
plt.ylabel('No. of Postive Cases')
plt.show()

-Visualizing male and female +ve cases data.
data = np.array(df[["positive","gender"]].groupby('gender').sum())
plt.pie([int(data[0][0]),int(data[1][0])], labels=["Female","Male"])
plt.legend()
plt.show()

-Visualizing male and female death cases data in a way that clearly shows the proportions of each category(male/female death ratio).
data = np.array(df[["death","gender"]].groupby('gender').sum())
plt.pie([int(data[0][0]),int(data[1][0])], labels=["Female","Male"])
plt.legend()
plt.show()

-show the no: of covid19 patients (x-axis) of respective regions.
df.health.value_counts().plot(kind='bar')

-create scatter plots in Matplotlib, determining whether two sets of (weather and +ve corona cases) data are correlated. If there is a correlation, scatter plots allow us to spot these trends.
plt.scatter(df.positive, df.Date)
plt.show()

-Using time series Matplotlib analysis the +ve cases data consists of data that contains dates. For example, in given dataset, plot Covid19 +ve over the last few weeks.
plt.plot(df.Date, df.positive)
plt.show()

----------------
Quiz # 03
#import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Q1. Read data from Q4_data.csv into the dataframe. (2 points)
df = pd.read_csv('Q4_data.csv')
df.head()

Q2. Display the description of numerical columns. (2 points)
df.describe()

Q3. Display information of all the columns. (1 point)
df.info()

Q4. Perform Correlation Analysis between all pairs of variables. Identify (by printing) what two input variables are highly
correlated to the output variable Final. (4 points)
df.corr()

Q5. Split the dataset into input and output variables. Then input the data into training and testing data set. Use 70% of the
data as training data and rest as testing data. Use 42 as random seed. (5 points)

x = df[['Quiz-1','Quiz-2','Quiz-3','Quiz-4','Major-1','Major-2','Major-3']]
y = df.Final

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

my_model = LinearRegression()

my_model.fit(x,y)

y_pred = my_model.predict(x)
print('Predicted Value:', y_pred)

my_model.score(x,y)

Q6. Using the scikit library function, apply Linear Regression between Major-1 and Final. Use all the data, print the
slope and the intercept. (Hint: Not to use the data from Q5). (6 points)


x = df['Major-1']
y = [df.Final]
from sklearn.linear_model import LinearRegression
LinReg  = LinearRegression()
LinReg.fit(x,y)
y_pred= LinReg.predict(x)
y_pred


--------LR----------

Linear Regression:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_csv
df.head()
df.plot.scatter(x='MID',y='FINAL')
df.corr()
x = df.MID
y = df.OVERALL
y = y.dropna(axis=0)

-------------- Code -----
def compute_coff(x,y):
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    SS_yy = np.sum(y*y) - n*m_y*m_y

    # calculating regression coefficients
    b = SS_xy / SS_xx
    a = m_y - b*m_x
    
    #calculating r and r_sq
    r = SS_xy/math.sqrt(SS_xx*SS_yy)
    r_sq = b*SS_xy/SS_yy
    
    return a,b,r,r_sq

a,b,r,r_squared = compute_coff(x,y)
print("Slope:",b)
print("Intercept:",a)
print("R:",r)
print("R Squared:",r_squared)


---------------------------Code---
---------Library----

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
marks = pd.read_csv('marks.csv')
marks.describe()

#separating dependent and independent variables
X = marks.MID.dropna()
y = marks.OVERALL.dropna()
#Data Preparation
X = X.values.reshape(-1,1)
y= y.values.reshape(-1,1)

# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x_train, y_train)
# Predict
y_predicted = regression_model.predict(x_test)

# model evaluation
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Mean squared error: ', mse)
print('Root mean squared error: ', mse**(1/2.0))
print('R2 score: ', r2)

-------Linear Regression-------
-----------D tree ----------


import pandas as pd
from scipy.stats import entropy

def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts() 
    print(p_data)
    # counts occurrence of each value
    entropy_ = entropy(p_data,base=2)  # get entropy from counts
    return entropy_

df = pd.read_csv('decision.csv')

entropy((4,2),base=2)

ent(df.playtennis)

-Gain:
def information_gain(df,column):
    var = df[column].value_counts()
    total=0
    for i in range(len(var)):
        total = total+ (var[i]/sum(var) * ent(df.playtennis[df[column]==var.index[i]]))
    return ent(df.playtennis)-total 

information_gain(df,'outlook')
information_gain(df,'temperature')
information_gain(df,'humidity')
information_gain(df,'wind')

#IG(Sunny/Temperature)
information_gain(df[df.outlook=='sunny'],'temperature')
#IG(Sunny/Humid)
information_gain(df[df.outlook=='sunny'],'humidity')
#IG(Sunny/wind)
information_gain(df[df.outlook=='sunny'],'wind')
#IG(Rain/Temp)
information_gain(df[df.outlook=='rain'],'temperature')
#IG(Rain/Humid)
information_gain(df[df.outlook=='rain'],'humidity')
#IG(Rain/wind)
information_gain(df[df.outlook=='rain'],'wind')

-----------D tree -----------
--------Logit and Clustering----------
import pandas as pd
import numpy as np
data = pd.read_csv("StudentGrades.csv")

data.columns

X = data[['age', 'sex', 'famsize', 'Pstatus', 'traveltime',
       'studytime', 'failures', 'internet', 'Dalc', 'Walc', 'health',
       'absences', 'G1', 'G2', 'G3']]
y = np.where(data.Total>=40,1,0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_Train, X_Test,y_train ,y_test = train_test_split(X,y,test_size=0.25)


clf = LogisticRegression(max_iter=500,random_state=0).fit(X_Train, y_train)
#clf.score(X_Test, y_test)

#clf.predict_proba(X[4:5])
y_pred = clf.predict(X_Test)

--------Model Evaluation--------
from sklearn.metrics import confusion_matrix
#y_true = [2, 0, 2, 2, 0, 1]
#y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_test, y_pred)

--------Clustering--------

x = [2,2,8,5,7,6,1,4]
y = [10,5,4,8,5,4,2,9]
import matplotlib.pyplot as plt
plt.scatter(x,y)

def distance(x1,y1,x2,y2):
  return ((x2-x1)**2+(y2-y1)**2)**(1/2.0)

distance(2,5,1.5,3.5)

---------Using Library-----------
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[2, 10], [2, 5], [8, 4],
               [5, 8], [7, 5], [6, 4],[1,2],[4,9]])

kmeans = KMeans(n_clusters=3, max_iter=1, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[6, 6], [3, 3]]))
print(kmeans.cluster_centers_)


import numpy as np
import matplotlib.pyplot as plt

#input matrix 
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
X

# creating the output array
y = np.array([[1], [1], [0]])

print("Actual Output:\n", y)

# shape of input array
print("\nShape of Output:", y.shape)

# We are using sigmoid as an activation function so defining the sigmoid function here

# defining the Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(sigmoid):
    return sigmoid * (1 - sigmoid)

# defining the model architecture
inputLayer_neurons = X.shape[1]  # number of features in data set
hiddenLayer_neurons = 3  # number of hidden layers neurons
outputLayer_neurons = 1  # number of neurons at output layer

# initializing weight
Weights_input_hidden = np.array([[0.42,0.88,0.55],[0.10,0.73,0.68],[0.60,0.18,0.47],[0.92,0.11,0.52]])

weights_hidden_output = np.array([0.30,0.25,0.23])

bh= np.array([0.46,0.72,0.08])

bout= np.array([0.69])


# defining the parameters
lr = 0.1
epochs = 1

losses = []
for epoch in range(epochs):
    ## Forward Propogation
    print('pass:',epoch)

# calculating hidden layer activations
np.dot(X, Weights_input_hidden)

hidden_layer_input= np.dot(X,Weights_input_hidden) + bh
hidden_layer_input

hiddenlayer_activations = sigmoid(hidden_layer_input)
hiddenlayer_activations


# calculating the output
output_layer_input = np.dot(hiddenlayer_activations , weights_hidden_output ) + bout
output = sigmoid(output_layer_input)
output

## Backward Propagation
# calculating error
E = y-output
E


# calculating rate of change of error w.r.t weight between hidden and output layer
Slope_output_layer= derivatives_sigmoid(output)
Slope_output_layer

Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
Slope_hidden_layer


# calculating rate of change of error w.r.t weights between input and hidden layer

d_output = E * Slope_output_layer*lr
d_output


# updating the weights
wout = np.array([[0.30],[0.25],[0.23]])

Error_at_hidden_layer = d_output*wout
Error_at_hidden_layer

d_hiddenlayer = Error_at_hidden_layer * Slope_hidden_layer
d_hiddenlayer

# print error at every 100th epoch
wout = wout + np.dot(hiddenlayer_activations.transpose(), d_output)*lr
wout


# appending the error of each epoch
wh =  Weights_input_hidden+ np.dot(X.transpose(),d_hiddenlayer)*lr
wh

------------NN BackPropagation-------------







---------------------------------