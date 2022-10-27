----------------------------------------------------------------
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

-------------Linear Regression-------------
-----------D tree ---------------------- 


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

-----------D tree ---------------------- 
-------------Logit and Clustering----------
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



---------Product imaging-------Keras -----------

# Importing the required Keras modules containing model and layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images.shape
train_images.shape
train_labels


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class_names[train_labels[3]]


def showimage(img,cmap='viridis'):

    plt.figure()
    plt.imshow(img,cmap=cmap)
    plt.grid(False)
    plt.show()

showimage(train_images[6],cmap='Blues')
train_images[0].shape

train_images[0]


#normalizing the data
train_images = train_images / 255.0

test_images = test_images / 255.0



train_images[0]


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Flatten(input_shape=(28, 28))) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.sigmoid))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest accuracy:', test_acc)

result = model.predict(np.expand_dims(test_images[3],0))
np.argmax(result)

#Building CNN Model
ntrain = train_images.shape[0]
ntest = test_images.shape[0]
input_shape = (28, 28, 1)

train_images = train_images.reshape(ntrain,28,28,1)
test_images = test_images.reshape(ntest,28,28,1)


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.predict(np.expand_dims(test_images[0],0))
------------- Keras End Here ---------------



NN BackPropogation
---------------------------------------------------


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

---------------------------------------------
-CNN vs Fully connected?
Convolutional layer is much more specialized, and efficient, than a fully connected layer. In a fully connected layer each neuron is connected to every neuron in the previous layer, and each connection has it's own weight.

-Differences between supervised and unsupervised learning?
Uses known and labeled data as input
Supervised learning has a feedback mechanism

Uses unlabeled data as input
Unsupervised learning has no feedback mechanism 

-How is logistic regression done?
Logistic regression measures the relationship between the dependent variable (our label of what we want to predict) and one or more independent variables (our features) by estimating probability using its underlying logistic function (sigmoid).

-steps in making a decision tree.
Take the entire data set as input
Calculate entropy of the target variable, as well as the predictor attributes
Calculate your information gain of all attributes (we gain information on sorting different objects from each other)
Choose the attribute with the highest information gain as the root node 
Repeat the same procedure on every branch until the decision node of each branch is finalized

Univariate
Univariate data contains only one variable. The purpose of the univariate analysis is to describe the data and find patterns that exist within it. 

Bivariate
Bivariate data involves two different variables. The analysis of this type of data deals with causes and relationships and the analysis is done to determine the relationship between the two variables.

Multivariate
Multivariate data involves three or more variables, it is categorized under multivariate. It is similar to a bivariate but contains more than one dependent variable.

-Given a data set consisting of variables with more than 30 percent missing values. How will you deal with them?
The following are ways to handle missing data values:
If the data set is large, we can just simply remove the rows with missing data values. It is the quickest way; we use the rest of the data to predict the values.
For smaller data sets, we can substitute missing values with the mean or average of the rest of the data using the pandas' data frame in python. There are different ways to do so, such as df.mean(), df.fillna(mean).

-Euclidean distance in Python?
The Euclidean distance can be calculated as follows:

euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

-Dimensionality reduction and its benefits?
The Dimensionality reduction refers to the process of converting a data set with vast dimensions into data with fewer dimensions (fields) to convey similar information concisely. 
This reduction helps in compressing data and reducing storage space. It also reduces computation time as fewer dimensions lead to less computing. It removes redundant features; for example, there's no point in storing a value in two different units (meters and inches). 

-Maintain a deployed model?
Monitor 
Constant monitoring of all models is needed to determine their performance accuracy. When you change something, you want to figure out how your changes are going to affect things. This needs to be monitored to ensure it's doing what it's supposed to do.

Evaluate
Evaluation metrics of the current model are calculated to determine if a new algorithm is needed. 

Compare
The new models are compared to each other to determine which model performs the best. 

Rebuild
The best performing model is re-built on the current state of data.

-Recommender systems?
Collaborative Filtering
As an example, Last.fm recommends tracks that other users with similar interests play often. This is also commonly seen on Amazon after making a purchase; customers may notice the following message accompanied by product recommendations: "Users who bought this also bought…"

Content-based Filtering
As an example: Pandora uses the properties of a song to recommend music with similar properties. Here, we look at content, instead of looking at who else is listening to music.

-outlier values be treated?
Try a different model. Data detected as outliers by linear models can be fit by nonlinear models. Therefore, be sure you are choosing the correct model.
Try normalizing the data. This way, the extreme data points are pulled to a similar range.
Drop the Outliers

-Accuracy = (True Positive + True Negative) / Total Observations

-You are given a dataset on cancer detection. You have built a classification model and achieved an accuracy of 96 percent. Why shouldn't you be happy with your model performance? What can you do about it?
Cancer detection results in imbalanced data. In an imbalanced dataset, accuracy should not be based as a measure of performance. It is important to focus on the remaining four percent, which represents the patients who were wrongly diagnosed. Early diagnosis is crucial when it comes to cancer detection, and can greatly improve a patient's prognosis.

-The K nearest neighbor algorithm can be used because it can compute the nearest neighbor and if it doesn't have a value, it just computes the nearest neighbor based on all the other features.

-The True Positive Rate (TPR) defines the probability that an actual positive will turn out to be positive. 
The True Positive Rate (TPR) is calculated by taking the ratio of the [True Positives (TP)] and [True Positive (TP) & False Negatives (FN) ]. 

The formula for the same is stated below -

TPR=TP/TP+FN

The False Positive Rate (FPR) defines the probability that an actual negative result will be shown as a positive one i.e the probability that a model will generate a false alarm. 
The False Positive Rate (FPR) is calculated by taking the ratio of the [False Positives (FP)] and [True Positives (TP) & False Positives(FP)].

The formula for the same is stated below -

FPR=FP/TP+FP

-What are the drawbacks of the linear model?
The assumption of linearity of the errors
It can't be used for count outcomes or binary outcomes
There are overfitting problems that it can't solve

-What is the difference between a box plot and a histogram?
The frequency of a certain feature’s values is denoted visually by both box plots and histograms. 

Boxplots are more often used in comparing several datasets and compared to histograms, take less space and contain fewer details. Histograms are used to know and understand the probability distribution underlying a dataset.

-NLP is short for Natural Language Processing. It deals with the study of how computers learn a massive amount of textual data through programming. A few popular examples of NLP are Stemming, Sentimental Analysis, Tokenization, removal of stop words, etc.

--------------------------------------------------------------------






















































