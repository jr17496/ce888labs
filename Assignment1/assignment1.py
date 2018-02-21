


from __future__ import print_function
#import keras
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
import os


#######################################################
#Cifar10 data load
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, shuffled and split between train and test sets:
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

#################################################################

from keras.datasets import mnist




# the data, shuffled and split between train and test sets
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()



#each image is 28x28
#x_train_mnist[i] row
#x_train_mnist[i][j]  j-th pixels in row i

import numpy as np
list_permutation_mnist_a = []
list_permutation_mnist_b = []
first_permu_mnist = []
second_permu_mnist = []
third_permu_mnist = []
four_permu = []
five_permu = []

for i in range(0,5):
    list_permutation_mnist_a.append(np.random.permutation(28))
    list_permutation_mnist_b.append(np.random.permutation(28))

for i in range(0,len(x_train_mnist)):
    first_permu_mnist.append(x_train_mnist[i][list_permutation_mnist_a[0]][list_permutation_mnist_b[0]])
    second_permu_mnist.append(x_train_mnist[i][list_permutation_mnist_a[1]][list_permutation_mnist_b[1]])
    third_permu_mnist.append(x_train_mnist[i][list_permutation_mnist_a[2]][list_permutation_mnist_b[2]])
a0 = np.asarray(x_train_mnist)
a1 = np.asarray(first_permu_mnist)
a2 = np.asarray(second_permu_mnist)
a3 = np.asarray(third_permu_mnist)
np.save('unchanged_mnist',a0)
np.save('first_permu_mnist',a1)
np.save('second_permu_mnist',a2)
np.save('third_permu_mnist',a3)
########################################################
####CIFAR10 dataset
####32x32 images

list_permutation_cifar10_a = []
list_permutation_cifar10_b = []
first_permu_cifar10 = []
second_permu_cifar10 = []
third_permu_cifar10 = []


for i in range(0,5):
    list_permutation_cifar10_a.append(np.random.permutation(32))
    list_permutation_cifar10_b.append(np.random.permutation(32))


for i in range(0,len(x_train_cifar10)):
    first_permu_cifar10.append(x_train_cifar10[i][list_permutation_cifar10_a[0]][list_permutation_cifar10_b[0]])
    second_permu_cifar10.append(x_train_cifar10[i][list_permutation_cifar10_a[1]][list_permutation_cifar10_b[1]])
    third_permu_cifar10.append(x_train_cifar10[i][list_permutation_cifar10_a[2]][list_permutation_cifar10_b[2]])
b0 = np.asarray(x_train_cifar10)
b1 = np.asarray(first_permu_cifar10)
b2 = np.asarray(second_permu_cifar10)
b3 = np.asarray(third_permu_cifar10)
np.save('unchanged_cifar10',b0)
np.save('first_permu_cifar10',b1)
np.save('second_permu_cifar10',b2)
np.save('third_permu_cifar10',b3)