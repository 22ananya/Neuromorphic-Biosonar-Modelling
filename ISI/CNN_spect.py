import keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import scipy.io as spio
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
num_classes = 1
epochs = 50

# input image dimensions
img_rows, img_cols = 35, 63

# load the data, split between train and test sets
X1 = spio.loadmat('spec_exfreq_sigs.mat', squeeze_me=True)['st_sp']
X1 = np.transpose(X1)
print(np.shape(X1))
X2 = spio.loadmat('spec_exfreq_sigs.mat', squeeze_me=True)['dy_sp']
X2 = np.transpose(X2)
X = np.append(X1, X2, axis = 0)


y1 = np.ones(len(X1))
y2 = np.zeros(len(X2))
y = np.append(y1,y2)
print(y.shape)
print(X.shape)


# Check data shape and format
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

print(X.shape)

#Slicing for testing
#X = X[0:4126,:,:,:]

print('X shape:', X.shape)
print('y shape:', y.shape)



# convert class vectors to binary class matrices
#y = keras.utils.to_categorical(y)
print(y[0])

#plot the first image in the dataset
print(X[0].shape)
plt.imshow(X[0].reshape(X.shape[1], X.shape[2]))
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25,random_state=21)

# Check if data was split correctly

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# CNN Model Build
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),strides=(1, 1),
                 activation='relu',
                 input_shape=(35,63,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(200, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

# CNN computation
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

#history = AccuracyHistory()

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split= 0.20,)
          #callbacks=[history])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
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
