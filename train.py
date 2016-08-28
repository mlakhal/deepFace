"""
===================================================
Faces recognition example CNN model 
===================================================
Train a simple deep CNN on the Wild face recognition dataset
                    small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_recognition.py


The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.92      0.85      0.88        13
     Colin Powell       0.91      0.97      0.94        60
  Donald Rumsfeld       0.96      0.81      0.88        27
    George W Bush       0.91      0.95      0.93       146
Gerhard Schroeder       0.87      0.52      0.65        25
      Hugo Chavez       0.57      0.87      0.68        15
       Tony Blair       0.88      0.83      0.86        36

      avg / total       0.89      0.89      0.88       322
================== ============ ======= ========== =======

"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils

from utils.data import reshapeDataset
from models import FaceCNN

#print(__doc__)

'''
  Hyperparameters of the CNN model
'''
#############################################################
BATCH_SIZE = 100
NB_EPOCH = 40
#############################################################
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = reshapeDataset( lfw_people.images, 32, 32)
h, w = 32, 32
print(X.shape)

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
nb_class = len(target_names)
''' 
Classes:
    0- Ariel Sharon
    1- Colin Powell
    2- Donald Rumsfeld
    3- George W Bush
    4- Gerhard Schroeder
    5- Hugo Chavez
    6- Tony Blair
'''


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_classes: %d" % nb_class)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_class)
Y_test = np_utils.to_categorical(y_test, nb_class)

#print X_train
X_train /= 255; X_test /= 255
###############################################################################
# Train our CNN classification model
model = FaceCNN(nb_class, 0.1, NB_EPOCH, BATCH_SIZE)

model.train(X_train, Y_train, X_test, Y_test)
###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = model.predict_classes(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(nb_class)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

plt.show()
# ploting the result
model.save_accuracy()
model.save_loss()
model.save_weights()
