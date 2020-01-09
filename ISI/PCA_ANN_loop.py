#Dependencies
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import scipy.io as spio
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf


#Import Data
# load the data, split between train and test sets
X = spio.loadmat('ISI_step20_GCK_binsize3_nIHC_CNN_data.mat', squeeze_me=True)['data']
y = spio.loadmat('ISI_step20_GCK_binsize3_nIHC_CNN_label.mat', squeeze_me=True)['label']


#shape before resizing
print(np.shape(X))
print(np.shape(y))

print(X[500,:,:])

#X = X[:,:,0:20]
print(np.shape(X))

print('cut shape is: ' , np.shape(X))
## Reshape correctly
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

print(np.shape(X))
print(X[500,:])

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20,random_state=21)


## Standardize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X[500,:])

#Setup PCA
from sklearn.decomposition import PCA
pca = PCA() #n_components=60

#Perform PCA
pca.fit(X_train)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()

pos_list = [50, 75, 100, 200, 500]
#pos_list = [10, 50]

ressies = []
pos_list.reverse()
for i in pos_list:

    pca = PCA(n_components=i)

    # Perform PCA
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    print(X_train.shape)

    sz = pca.n_components_
    #print('PCA Components: ', sz)

    # Neural network
    model = Sequential()
    model.add(Dense(16, input_dim=sz, activation= 'relu'))
    #model.add(Dropout(0.35))]
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dropout(0.35))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation= 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70)
    mc = ModelCheckpoint('best_ANN_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=256, validation_split=0.20, epochs=500, verbose=0, callbacks=[es, mc])


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy with' + str(i) +  ' principal components')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['DR Train', 'DR Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # load the saved model
    saved_model = load_model('best_ANN_model.h5')
    # evaluate the model
    train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    print(train_acc, test_acc)
    ressies.append(test_acc)

print(ressies)

x = np.arange(len(pos_list))
bar_p = []
for i in ressies:
    bar_p.append(i[1])

plt.bar(x,bar_p)
plt.xticks(x, ('2000', '1000', '500', '200', '100', '75', '50'))
plt.title('Results')
plt.ylabel('Accuracy')
plt.xlabel('nComponents')
plt.show()


from sklearn.metrics import roc_curve
X_test = tf.cast(X_test, tf.float32)
y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

#print("rf model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_keras) * 100)

#y_pred_test = bnb.predict(X_test)
#print("Naive_Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_test) * 100)
#joblib.dump(cv.best_estimator_, '../../rf_cv.pkl')
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='AUC  (area = {:.3f})'.format(auc_keras))
plt.xlabel('False Alarm')
plt.ylabel('Hit')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
