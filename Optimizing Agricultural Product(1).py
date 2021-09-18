#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Step 1 : Import the libraries

#For manupulation
import numpy as np
import pandas as pd

#For data visualization
import matplotlib.pyplot as plt
import seaborn as sns


#For interactivity
from ipywidgets import interact


# In[2]:


#Step 2 : Upload the dataset
    
data = pd.read_csv("data.csv")


# In[3]:


#Lets check the shape of the dataset

print("shape of the dataset:",data.shape )


# In[4]:


data.head()  # First five row of dataset


# In[5]:


#step 3 : Data Preprocessing and Exploratary analysis of data

data.isnull().sum()   # check for missing value in data


# > fillna() functionn is used to replace the missing value with statical value such as Mean,Median or Mode
# 
# >NA means not available
# 
# >pandas have function like fillna(),dropna() to treat missing values.

# In[6]:


#lets check the crop present in data set

data['label'].value_counts()


# In[7]:


#let's check summary of all crops

print('Average ratio of Nitrogen in soil:{0:.2f}'.format(data['N'].mean()))
print('Average ratio of phousphorus in soil:{0:.2f}'.format(data['P'].mean()))
print('Average ratio of Potassium in soil:{0:.2f}'.format(data['K'].mean()))
print('Average ratio of temprature in celsius:{0:.2f}'.format(data['temperature'].mean()))
print('Average Relative Humidity in % :{0:.2f}'.format(data['humidity'].mean()))
print('Average PH Value of soil:{0:.2f}'.format(data['ph'].mean()))
print('Average rainfall in mm : {0:.2f}'.format(data['rainfall'].mean()))


# In[8]:


#Let's check the summary statistics for each of the crops

@interact

def summary(crops = list(data['label'].value_counts().index)):
    X = data[data['label'] == crops]
    print("-------------------------------------------")
    
    print("statistics for Nitrogen")
    print("Minimum Nitrogen required :", X['N'].min())
    print("Average Nitrogen required :", X['N'].mean())
    print("Maximum Nitrogen required :", X['N'].max())
    print("------------------------------------------")
    
    print("statistics for Phosphorus")
    print("Minimum Phosphorus required :", X['P'].min())
    print("Average Phosphorus required :", X['P'].mean())
    print("Maximum Phosphorus required :", X['P'].max())
    print("---------------------------------------------")
    
    print("statistics for Potassium")
    print("Minimum Potassium required :", X['K'].min())
    print("Average Potassium required :", X['K'].mean())
    print("Maximum Potassium required :", X['K'].max())
    print("----------------------------------------------")
    
    print("statistics of temperature")
    print("Minimum temperature required :{0:.2f}".format(X['temperature'].min()))
    print("Average temperature required :{0:.2f}".format(X['temperature'].mean()))
    print("Maximum temperature required :{0:.2f}".format(X['temperature'].max()))
    print("------------------------------------------------")
    
    print("statistics of Humidity")
    print("Minimum humidity required :{0:.2f}".format(X['humidity'].min()))
    print("Average humidity required :{0:.2f}".format(X['humidity'].mean()))
    print("Maximum humidity required :{0:.2f}".format(X['humidity'].max()))
    print("------------------------------------------------")
    
    print("statistics of PH")
    print("Minimum PH required :{0:.2f}".format(X['ph'].min()))
    print("Average PH required :{0:.2f}".format(X['ph'].mean()))
    print("Maximum PH required :{0:.2f}".format(X['ph'].max()))
    print("------------------------------------------------")
    
    print("statistics of Rainfall")
    print("Minimum rainfall required :{0:.2f}".format(X['rainfall'].min()))
    print("Average rainfall required :{0:.2f}".format(X['rainfall'].mean()))
    print("Maximum rainfall required :{0:.2f}".format(X['rainfall'].max()))
    print("------------------------------------------------")
    
    
    
    


# In[9]:


#Let's compare the average requirement for each crop with average  conditions

@interact
def compare(conditions = ['N' , 'P' , 'K' , 'temperatue' , 'PH' , 'humidity' , 'rainfall']):
    print("average value for",conditions,"is{0:.2f}".format(data[conditions].mean()))
    print("-----------------------------------------------------------------------")
    
    print("Rice : {0:.2f}".format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Black Geams : {0:.2f}".format(data[(data['label'] == 'blackgram')][conditions].mean()))
    print("Banana : {0:.2f}".format(data[(data['label'] == 'banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(data[(data['label'] == 'jute')][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(data[(data['label'] == 'apple')][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[(data['label'] == 'muskmelon')][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[(data['label'] == 'watermelon')][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(data[(data['label'] == 'kidneybeans')][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[(data['label'] == 'mungbean')][conditions].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(data[(data['label'] == 'Chickpea')][conditions].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[(data['label'] == 'cotton')][conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[(data['label'] == 'mothbeans')][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(data[data['label'] == 'mango'][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(data[(data['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(data[(data['label'] == 'coffee')][conditions].mean()))
                                         
                                          
                                          
                                          
                                          
                                          


# In[10]:


#Let's make this function more intuative

@interact
def compare(conditions = ['N' , 'P' , 'K' , 'temperature' , 'ph' , 'humidity' , 'rainfall']):
    print("Crops which require greater then average",conditions,'\n')
    print(data[data[conditions] > data[conditions].mean()]['label'].unique())
    print("---------------------------------------------------------------------")
    
    print("Crops which require less than average",conditions,"\n")
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())
    


# In[11]:


#Step 4 : Visualize the dataset 

plt.subplot(2,4,1)
sns.distplot(data['N'],color = 'red')
plt.xlabel("Ratio of Nitrogen" , fontsize = 6)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['K'],color = 'black')
plt.xlabel("ratio of potassium" , fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['P'],color = 'Yellow')
plt.xlabel("ratio of phosphorus" , fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['temperature'],color = 'green')
plt.xlabel("ratio of temperature" , fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['rainfall'],color = 'grey')
plt.xlabel("ratio of rainfall" , fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['humidity'],color = 'lightgreen')
plt.xlabel("ratio of humidity" , fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['ph'],color = 'darkgreen')
plt.xlabel("ratio of ph" , fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agriculture conditions' ,fontsize = 20)
plt.show()


# In[12]:


#Let's find out some interesting facts

print("some Interesting Patterns")
print("-------------------------------------")

print("Crops with requires very high ratio of Nitrogen content in soil :", data[data['N'] > 120]['label'].unique())
print("Crops with requires very high ratio of Phosphorus content in soil :", data[data['P'] > 100]['label'].unique())
print("Crops with requires very high ratio of POtassium content in soil :", data[data['K'] > 200]['label'].unique())
print("Crops with requires very high rainfall :", data[data['rainfall'] > 200]['label'].unique())
print("Crops with requires very high Temprature :", data[data['temperature'] > 40]['label'].unique())
print("Crops with requires very low Temprature :", data[data['temperature'] < 10]['label'].unique())
print("Crops with requires very low humidity :", data[data['humidity'] > 20]['label'].unique())
print("Crops with requires very low PH :", data[data['ph'] < 4]['label'].unique())
print("Crops with requires very high PH :", data[data['ph'] > 9]['label'].unique())


# In[13]:


#Let's understand now which crops can only grow in summer season , winter season , and Rainy Season

print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("-----------------------------------")


print("winter crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("--------------------------------------------------")
           
print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())


# In[14]:


# we are going to do some clustering analysis

from sklearn.cluster import KMeans

#removing the label column
X = data.drop(['label'],axis=1)

#selecting all the value of the data

X = X.values

#checking the shape

print(X.shape)


# In[15]:


#Let's determine optimal number of cluster with in the Dataset

plt.rcParams['figure.figsize'] = (10,4)
wcss = []
for i in range(1 , 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300 , n_init = 10,random_state = 0)
    km.fit(X)
    wcss.append(km.inertia_)
    
#Let's plot the results

plt.plot(range(1 , 11),wcss)
plt.title("The Elbow Method",fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[16]:


#let's implement the k_Means algorithm to perform clustering analysis

km = KMeans(n_clusters = 4, init = 'k-means++' , max_iter = 300, n_init = 10 , random_state  = 0)
y_means = km.fit_predict(X)


# In[17]:


#Lets Find out the result

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0:'cluster'})


# In[18]:


#Lets check the cluster of each crops

print('Lets check the result after applying the kMeans clustering analysis \n')

print("Crops in First Cluster:",z[z['cluster']==0]['label'].unique())

print("----------------------------------------------------------------")

print("Crops in Second Cluster:",z[z['cluster']==1]['label'].unique())

print("--------------------------------------------------------")

print("Crops in Third Cluster:",z[z['cluster']==2]['label'].unique())

print("----------------------------------------------------------------")

print("Crops in Forth Cluster:",z[z['cluster']==3]['label'].unique())



# In[19]:


#Let's apply Predictive mode

#first split the the dataset

y = data['label']
x = data.drop(['label'], axis = 1)


print("Shape Of x :",x.shape)
print("shape of y :",y.shape)


# In[20]:


#Let's create training and testing set to validate the score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2, random_state = 0)



# In[21]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[22]:


#let's create a predictive model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)


# In[23]:


#Now Let's evaluate the Model Performance
from sklearn.metrics import confusion_matrix

#Let's print the confudion matrix first
plt.rcParams['figure.figsize'] = (10 , 10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm ,annot = True , cmap = 'Wistia')
plt.title("Confusion Metrix for Logistic Regression",fontsize = 15)
plt.show()


# In[24]:


#Let's print the classification report also
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


# In[25]:


#Let's check first five row
data.head()


# In[26]:


#Now predict the output agains input

prediction = model.predict((np.array([[90,
                                      40,
                                      42,
                                      20,
                                      80,
                                      7,
                                      204]])))

print("The suggested crop for Given Climetic conditions is :",prediction)


# In[27]:


#Another prediction
data.tail()


# In[28]:


prediction1 = model.predict((np.array([[100,
                                      30,
                                      31,
                                      25,
                                      62,
                                      6.8,
                                      160]])))

print("The suggested crop for Given Climetic conditions is :",prediction1)


# In[29]:


#Wow Great................

