# Building a ANN

# Data Preprocessing
#%%
# Import the libaries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#%%
# Import the data
dataset = pd.read_csv("./Skyserver_SQL2_27_2018 6_51_39 PM.csv")


X = dataset[['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol',
'field', 'specobjid', 'redshift', 'plate', 'mjd', 'fiberid']]
y = dataset['class']

#%%
# Encode Label
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()

y = encoder.fit_transform(y)

#%%
# Normalize Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

#%%
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

#%%
# Import keras libraries
import keras
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization


# K-fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # Building the model
    model = Sequential([
    Dense(10, activation='relu', init='uniform', input_shape=(17,)),
    Dropout(0.10),
    Dense(5, activation='relu', init='uniform'),
    Dropout(0.10),
    Dense(3, activation='softmax', init='uniform')
    ])
    # Compile
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_classifier, batch_size=32, nb_epoch=250)
accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10, n_jobs=-1)

print('Accuracies:', accuracies)
mean = accuracies.mean()
print('Mean:', mean)
variance = accuracies.std()
print('Variance:', variance)


