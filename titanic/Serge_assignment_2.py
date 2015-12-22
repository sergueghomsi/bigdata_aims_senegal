#################  ghomsi.k.serge@aims-senegal.org ###################
##############            Big Data Course               ##############
#########             Analysis of Titanic data              ##########
################  AIMS-Senegal December 2015    ######################

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import StringIO, pydot 
from sklearn.ensemble import RandomForestClassifier
from IPython.core.display import Image 
from sklearn import tree
import IPython


#######################################################################
##                            Training part        #######################################################################


# Define a patth of file and check if the files exist

training_data_filepath='/media/ghomsi/SERGUEI/Big_Data/bigdata_aims_senegal/titanic/data/train.csv'
test_data_filepath='/media/ghomsi/SERGUEI/Big_Data/bigdata_aims_senegal/titanic/data/test.csv'
print 'The path to the training data set is correct: ', os.path.exists(training_data_filepath)
print 'The path to the test data set is correct: ', os.path.exists(test_data_filepath)

# Reading of data

df_training = pd.read_csv(training_data_filepath)

# Dorp the columns which get missing data
df_training_clean = df_training.drop(['Age','Cabin','Embarked','Name','Sex','Ticket'],axis=1)

# drop of the Survived column from the data frame and assigns it to a variable target

target = df_training_clean.pop('Survived')

# plot of kde : Kernel density estimation (plot of probability density)  

#%matplotlib inline
df_training_clean['Fare'].plot(kind='kde')

array_training_clean=df_training_clean.values
array_target=target.values

#Intitialize the DecisionTreeClassifier algorithm with optional parameters 
classifier = tree.DecisionTreeClassifier(min_samples_leaf=5, max_depth=4) 
#use the initialized tree to learn the relationship between 
#known passanger information, features, and know state of survival 
classifier.fit(array_training_clean, array_target)

# We can see the score, the efficiency of the training as follows
print("Training Score: ", classifier.score(array_training_clean, array_target))


dot_data = StringIO.StringIO() 
tree.export_graphviz(classifier, out_file=dot_data, feature_names=df_training.keys()) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("decision_tree_ass1.png")

Image(filename='decision_tree_ass1.png') 

################################################################
###########     Weekend Assignment: Test part ##################
################################################################

# 1 load the test data into pandas DataFrame 
df_test = pd.read_csv(test_data_filepath)


# 2 Fill of missing data for the Fare column for one passenger.
mean_fare_by_pclass = df_test[['Pclass','Fare']].groupby(['Pclass']).agg('mean')['Fare']

fare_given_pclass=df_test['Pclass'].apply(lambda x: mean_fare_by_pclass[x])


#3 Drop of some features
df_test_clean = df_test.drop(['Age','Cabin','Embarked','Name','Sex','Ticket'],axis=1)

df_test['Fare'].fillna(fare_given_pclass,inplace=True)


# 4 Convert the test DataFrame to numpy array 
 array_test_clean = df_test_clean.values



# 5 Predict the survival of passengers using the tree we trained as
survival_prediction = classifier.predict(array_test_clean)    

# 6 Save the prediction to a file. If you want, you can submit this file in Kaggle page and get your score

 with open('prediction_serge.csv', 'w') as csvfile:    
     for survived in survival_prediction:
          csvfile.write("{}\n".format(survived)) 

#7 BONUS: Visualize the result:

print survival_prediction.shape
fig, ax = plt.subplots()
ax.set_xticklabels(df_test_clean['PassengerId'])
plt.plot(survival_prediction,linestyle='',marker='x',color='b')
plt.title("Survival prediction")
plt.xlabel("Passenger Identifier")
plt.text(300, 0.5,'0: Deid')
plt.text(300, 0.7,'1: Survived')
 

# #Compare predictions from tree vs forest 

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(array_training_clean,array_target)
output = model.predict(array_test_clean)


# # visualisation of comparaison
print output.shape
print survival_prediction.shape
fig, ax = plt.subplots()
ax.set_xticklabels(df_test_clean['PassengerId'])
plt.plot(survival_prediction-output,linestyle='',marker='x',color='b')
plt.title("Comparaison from tree Vs forest")
plt.xlabel("Passenger Identifier")
plt.text(300, 0.5,'0: tree = forest')
plt.text(300, 0.7,'-1 or 1: tree != forest')
 

#######################################################################
####################  end program #####################################
#######################################################################



