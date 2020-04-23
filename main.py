""" 
Python Version 3.6.5

"""

import numpy as np
import pandas as pd

# %%Preprocessing-steps
"""
loading the data into pandas data frame
"""
data = pd.read_csv("glassdata.csv")

"""
To see the stats of the data set excluding the labels(Glass_Type)
"""
data.Glass_Type = pd.Categorical(data.Glass_Type)
data.describe(exclude=['category'])

"""
Normalizing the data set and replacing Glass_Type lables with their categorical index
"""
data.iloc[:, 0:10] = data.iloc[:, 0:10].apply(lambda x: x/x.max(), axis=0)
data['labels'] = data.Glass_Type.cat.codes
data.head()

"""
See the count of each class
"""
data.Glass_Type.value_counts()

"""
splitting the glass dataset into 8:2 train/test ratio
"""
data_X = data.iloc[:, 0:10]
data_Y = data['labels']
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=17)

#%% parameters and initializing the network

import BP_ANN
#import matplotlib.pyplot as plt
layers = [10, 6, 6]
learning_rate = 0.02
momentum = 0.0001
regularization = 0.5


model = BP_ANN.Backprop_ANN(layers, learning_rate, momentum , regularization)
initial_weights = model.weights.copy()
initial_biases = model.biases.copy()

#%% training the network

acc = model.network_train(train_X.as_matrix(), train_Y.as_matrix(), test_X.as_matrix(), test_Y.as_matrix(), No_epochs=179)

##plt.figure()
##plt.title('Validation Accuracy per Training Epoch')
##plt.xlabel('epoch')
##plt.ylabel('prediction accuracy')
##plt.plot(acc)
##plt.show()
accuracy = max(acc)
print("Accuracy: %s" % (accuracy))

#%% predection and stats 

##import seaborn as sns
from sklearn.metrics import confusion_matrix
predictions = model.network_test(test_X.as_matrix())
labels = ['1', '2', '3', '5', '6', '7']
cm = confusion_matrix(test_Y.as_matrix(), predictions)
cm = pd.DataFrame(cm, index=labels, columns=labels)
cm.to_csv('cm.csv', header=True, index=True)
##cm_plot = sns.heatmap(cm, annot=True, cmap="binary")

##plt.title("Confusion Matrix")
##plt.xlabel("Predicted Class")
##plt.ylabel("True Class")
##plt.show()


from BP_ANN import precision_recall 
stats = precision_recall(cm.as_matrix())
stats1 = pd.DataFrame(stats).transpose()
stats_csv = pd.DataFrame(stats1)
stats_csv.columns = ['1', '2', '3', '5', '6', '7']
stats_csv.to_csv('PRE_REC.csv', header=True, index=True)
for q, s in zip([1, 2, 3, 5, 6, 7], stats):
    print(f'Glass_Type: {q}: {s}')
    
#%% Actual and predicted output
labels = np.hstack([test_Y.as_matrix().reshape(test_Y.shape[0], 1), np.array(predictions).reshape(len(predictions), 1)])
labels = pd.DataFrame(labels, columns=["Actual", "Predicted"])
labels.to_csv('lables.csv', header=True, index=True)

#%% initial and final weight vectors

final_weights_ih = model.weights[0] # between input and hidden
final_weights_ho = model.weights[1] # between hidden and output
finalih = pd.DataFrame.from_records(final_weights_ih)
finalho = pd.DataFrame.from_records(final_weights_ho)
finalih.to_csv('finalweightsih.csv', header=True, index=True)
finalho.to_csv('finalweightsho.csv', header=True, index=True)


initial_weights_ih = initial_weights[0] # between input and hidden
initial_weights_ho = initial_weights[1] # between hidden and output
initialih = pd.DataFrame.from_records(initial_weights_ih)
initialho = pd.DataFrame.from_records(initial_weights_ho)
initialih.to_csv('initialweightsih.csv', header=True, index=True)
initialho.to_csv('initialweightsho.csv', header=True, index=True)    
