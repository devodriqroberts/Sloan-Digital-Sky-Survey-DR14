# Building a ANN

# Data Preprocessing
#%%
# Import the libaries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#%%
# Import the data
dataset = pd.read_csv("./Churn_Modelling.csv")

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]


#%%
# Encode Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
onehot = OneHotEncoder(categorical_features=[2])
cat_cols = ['Geography', 'Gender']
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col])
X = onehot.fit_transform(X).toarray()

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
# # Import keras libraries
# import keras
# import keras.backend as K
# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# from keras.layers import BatchNormalization

# # Building the model
# model = Sequential([
#     Dense(6, activation='relu', init='uniform', input_shape=(11,)),
#     Dropout(0.10),
#     Dense(6, activation='relu', init='uniform'),
#     Dropout(0.10),
#     Dense(1, activation='sigmoid', init='uniform')
# ])

# # Compile
# model.compile(optimizer='adam', 
# loss='binary_crossentropy', 
# metrics=['accuracy'])

# # Fit model
# model.fit(X_train, y_train, 
# epochs=100, 
# batch_size=10, 
# validation_split=0.2,
# verbose=0)

# # Predict
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)
# y_pred[:10]

# # Confusion matix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)


# # K-fold Cross Validation
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

# def build_classifier():
#     # Building the model
#     model = Sequential([
#     Dense(6, activation='relu', init='uniform', input_shape=(11,)),
#     Dropout(0.10),
#     Dense(6, activation='relu', init='uniform'),
#     Dropout(0.10),
#     Dense(1, activation='sigmoid', init='uniform')
#     ])
#     # Compile
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# model = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
# accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10, n_jobs=-1)

# print('Accuracies:', accuracies)
# mean = accuracies.mean()
# print('Mean:', mean)
# variance = accuracies.std()
# print('Variance:', variance)

# Tuning with grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization

def build_classifier(optimizer):
    # Building the model
    model = Sequential([
    Dense(6, activation='relu', init='uniform', input_shape=(11,)),
    Dropout(0.10),
    Dense(6, activation='relu', init='uniform'),
    Dropout(0.10),
    Dense(1, activation='sigmoid', init='uniform')
    ])
    # Compile
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_classifier)

parameters = {
    'batch_size' : [25,32], 
    'nb_epoch' : [100, 250],
    'optimizer' : ['adam', 'rmsprop']
    }

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters, 
    cv=10, 
    n_jobs=-1,
    scoring='accuracy')

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

