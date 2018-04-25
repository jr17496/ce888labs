from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import Callback
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import random
from sklearn.metrics import f1_score
#############################################################
#Assignment2
#############################################################


input_shape = 784
#Autoencoder model
def instantiate_autoencoder_and_classifier(input_shape):
    input_img = Input(shape=(input_shape,))
    encoded = Dense(24, activation='relu')(input_img)
    decoded = Dense(input_shape, activation='sigmoid')(encoded)
    x = Dense(20, activation='relu')(encoded)
    output = Dense(10,activation = 'softmax')(x)
    autoencoder = Model(input_img, decoded)
    classifier = Model(input_img,output)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    classifier.compile(optimizer='adam',loss = 'binary_crossentropy')
    return autoencoder, classifier






num_classes = 10

x_train_mnist = np.load('unchanged_mnist.npy')
x_first_permu_mnist_train = np.load('first_permu_mnist.npy')
x_second_permu_mnist_train = np.load('second_permu_mnist.npy')
x_third_permu_mnist_train = np.load('third_permu_mnist.npy')

#####################
#Split train set input into training and validation
#####################

lenght_x_train = len(x_train_mnist)

validation_split = int(lenght_x_train * 0.8)



x_train_validation_mnist = x_train_mnist[validation_split:,:]
x_train_mnist= x_train_mnist[:validation_split,:]


x_first_permu_validation_mnist = x_first_permu_mnist_train[validation_split:,:]
x_first_permu_mnist_train= x_first_permu_mnist_train[:validation_split,:]


x_second_permu_validation_mnist = x_second_permu_mnist_train[validation_split:,:]
x_second_permu_mnist_train = x_second_permu_mnist_train [:validation_split,:]

x_third_permu_validation_mnist = x_third_permu_mnist_train[validation_split:,:]
x_third_permu_mnist_train = x_third_permu_mnist_train [:validation_split,:]

x_validation_mnist = np.concatenate((x_train_validation_mnist,x_first_permu_validation_mnist),axis = 0)
x_validation_mnist = np.concatenate((x_validation_mnist,x_second_permu_validation_mnist),axis = 0)
x_validation_mnist = np.concatenate((x_validation_mnist,x_third_permu_validation_mnist),axis = 0)

#####################




x_train_mnist = np.concatenate((x_train_mnist,x_first_permu_mnist_train),axis = 0)
x_train_mnist = np.concatenate((x_train_mnist,x_second_permu_mnist_train),axis = 0)
x_train_mnist = np.concatenate((x_train_mnist,x_third_permu_mnist_train),axis = 0)




y_train_mnist = np.load('targets_mnist_train.npy')

y_validation_mnist = y_train_mnist[validation_split:]
y_train_mnist = y_train_mnist[:validation_split]

y_train_mnist = np.concatenate((y_train_mnist,y_train_mnist),axis = 0)
y_train_mnist = np.concatenate((y_train_mnist,y_train_mnist),axis = 0)


y_validation_mnist = np.concatenate((y_validation_mnist,y_validation_mnist),axis = 0)
y_validation_mnist = np.concatenate((y_validation_mnist,y_validation_mnist),axis = 0)
#x_train_mnist = np.hsplit(x_combined_train, 2)


set_of_labels = set(y_validation_mnist)
labels = list(set_of_labels)

y_train_mnist = np_utils.to_categorical(y_train_mnist, num_classes)

y_validation_mnist = np_utils.to_categorical(y_validation_mnist, num_classes)




y_test_mnist = np.load('targets_mnist_test.npy')
y_test_mnist = np_utils.to_categorical(y_test_mnist, num_classes)


x_train_mnist = x_train_mnist.astype('float32') / 255.
x_validation_mnist = x_validation_mnist.astype('float32') / 255.

x_train_mnist = x_train_mnist.reshape((len(x_train_mnist), np.prod(x_train_mnist.shape[1:])))
x_validation_mnist = x_validation_mnist.reshape((len(x_validation_mnist), np.prod(x_validation_mnist.shape[1:])))


input_shape = 784

n = 1

#####################
##Initialize n classifiers
#####################

for i in range(0,n):
    autoencoder,classifier = instantiate_autoencoder_and_classifier(input_shape)
    autoencoder.save('autoencoder_plus'+str(i)+'.h5')
    classifier.save('classifier_plus' + str(i) + '.h5')


#############################################
#Pretrain each of the autoencoders on a small portion of the training set
#############################################

samplesize = 2000

print('Pretraining autoencoders')

for i in range(0,n):

    filepath_autoencoder = "weights_autoencoder_plus.best.hdf5"
    checkpoint_autoencoder = ModelCheckpoint(filepath_autoencoder, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop_autoencoder = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    callbacks_list_autoencoder = [checkpoint_autoencoder, early_stop_autoencoder]

    autoencoder = load_model('autoencoder_plus'+str(i)+'.h5')

    random_indexes = random.sample(range(0, len(x_train_mnist)), samplesize)
    training_sample = x_train_mnist[random_indexes,:]
    autoencoder.fit(training_sample, training_sample,
                    epochs=1,
                    batch_size=1,
                    shuffle=True,
                    verbose = False
                  #  validation_data=(x_validation_mnist, x_validation_mnist),
                  #  callbacks=callbacks_list_autoencoder
                    )




batchsize = 10000

score_for_autoencoder = []
print('Training')
epochs = 5
f1_previous = 0
f1_total = 0
number_of_validation_batches = 20
validation_batch_size = int(len(x_validation_mnist)/number_of_validation_batches)
number_of_batches = int(len(x_train_mnist)/batchsize)
initial_threshold_for_classifier_creation =0.7
threshold_classifier_creation = initial_threshold_for_classifier_creation
for k in range(0,epochs):
    print(k)
    for i in range(0,number_of_batches):
        start_index = i * batchsize
        end_index = batchsize+(i *batchsize)
        train_set = x_train_mnist[start_index:end_index,:]
        target_set = y_train_mnist[start_index:end_index,:]
        for j in range(0,n):
            autoencoder = load_model('autoencoder_plus'+str(j)+'.h5')
            score_for_autoencoder.append(autoencoder.evaluate(train_set, train_set,verbose= False))
        #print(np.max(score_for_autoencoder))
        print(np.min(score_for_autoencoder))
        #print(np.max(score_for_autoencoder)-np.min(score_for_autoencoder))
        #print(np.min(score_for_autoencoder)*0.05)
        if threshold_classifier_creation > np.min(score_for_autoencoder):
            min_error_index = np.argmin(score_for_autoencoder)
            threshold_classifier_creation = 0.15
            autoencoder = load_model('autoencoder_plus' + str(min_error_index) + '.h5')
            classifier = load_model('classifier_plus' + str(min_error_index) + '.h5')
            index_increased = False
        else:
            print('Classifier was created')
            autoencoder, classifier = instantiate_autoencoder_and_classifier(input_shape)
            autoencoder.save('autoencoder_plus' + str(n) + '.h5')
            classifier.save('classifier_plus' + str(n) + '.h5')
            n = n+1
            threshold_classifier_creation = 0.7
            index_increased = True

        score_for_autoencoder = []
        autoencoder.fit(train_set,train_set,
            epochs=1,
            batch_size=1,
            shuffle=True,
            verbose = False
            #validation_data=(x_validation_mnist, x_validation_mnist),
                    )
        classifier.fit(train_set, target_set,
                    epochs=1,
                    batch_size=1,
                    shuffle=True,
                       verbose=False
            #        validation_data=(x_validation_mnist, y_validation_mnist),
                   )

        if index_increased == True:
            autoencoder.save('autoencoder_plus' + str(n) + '.h5')
            classifier.save('classifier_plus' + str(n) + '.h5')
        else:
            autoencoder.save('autoencoder_plus' + str(min_error_index) + '.h5')
            classifier.save('classifier_plus' + str(min_error_index) + '.h5')

        predictions = classifier.predict(train_set)
    for l in range(0, number_of_validation_batches):
        start_index = l * validation_batch_size
        end_index = validation_batch_size + (l * validation_batch_size)

        test_set = x_validation_mnist[start_index:end_index, :]
        target_set = y_validation_mnist[start_index:end_index, :]

        for j in range(0, n):
            autoencoder = load_model('autoencoder_plus' + str(j) + '.h5')
            score_for_autoencoder.append(autoencoder.evaluate(test_set, test_set))

        min_error_index = np.argmin(score_for_autoencoder)
        classifier = load_model('classifier_plus' + str(min_error_index) + '.h5')
        predictions = classifier.predict(test_set)
        f1 = f1_score(target_set, predictions.round(), labels=labels, average='weighted')
        f1_total = f1_total +f1

        score_for_autoencoder = []
    f1_average = f1_total/number_of_validation_batches
    f1_total = 0
    if (f1_previous< f1_average):
        f1_previous = f1_average
    else:
        k = epochs-1



