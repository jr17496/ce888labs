from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import Callback
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import random


num_classes = 10


number_of_classifiers = 10




x_test_cifar_10 = np.load('unchanged_cifar10_test.npy')
x_first_permu_cifar_10_test = np.load('first_permu_cifar10_test.npy')
x_second_permu_cifar_10_test = np.load('second_permu_cifar10_test.npy')
x_third_permu_cifar_10_test = np.load('third_permu_cifar10_test.npy')
x_test_cifar_10 = np.concatenate((x_test_cifar_10,x_first_permu_cifar_10_test,x_second_permu_cifar_10_test,x_third_permu_cifar_10_test),0)
x_test_cifar_10 = x_test_cifar_10.astype('float32') / 255.
x_test_cifar_10 = x_test_cifar_10.reshape((len(x_test_cifar_10), np.prod(x_test_cifar_10.shape[1:])))



y_test_cifar_10 = np.load('targets_cifar10_test.npy')
y_test_cifar_10 = np.concatenate((y_test_cifar_10,y_test_cifar_10),axis = 0)
y_test_cifar_10 = np.concatenate((y_test_cifar_10,y_test_cifar_10),axis = 0)

y_test_cifar_10_list = []
for i in range(0,len(y_test_cifar_10)):
    y_test_cifar_10_list.append(y_test_cifar_10[i][0])


set_of_labels = set(y_test_cifar_10_list)
labels = list(set_of_labels)

y_test_cifar_10 = np_utils.to_categorical(y_test_cifar_10_list, num_classes)


print(len(y_test_cifar_10))
print(len(x_test_cifar_10))

number_of_batches = 20
batch_size = int(len(x_test_cifar_10)/number_of_batches)

score_for_autoencoder = []
list_of_predictions = []


statistics = []
statistics_for_batch = []

total_precision = 0
total_recall = 0
total_accuracy = 0
total_f1 = 0


for i in range(0,number_of_batches):
    #print(i)
    start_index = i * batch_size
    end_index = batch_size + (i * batch_size)
    test_set = x_test_cifar_10[start_index:end_index, :]
    target_set = y_test_cifar_10[start_index:end_index, :]

    for j in range(0,number_of_classifiers):
        autoencoder = load_model('autoencoder_plus_cifar' + str(j) + '.h5')
        score_for_autoencoder.append(autoencoder.evaluate(test_set, test_set,verbose=0))

    min_error_index = np.argmin(score_for_autoencoder)
    print(min_error_index)
    classifier = load_model('classifier_plus_cifar' + str(min_error_index) + '.h5')
    predictions = classifier.predict(test_set)
    for m in range(len(predictions)):
        for n in range(len(predictions[m])):
            if (predictions[m][n] == max(predictions[m])):
                predictions[m][n] =1
            else:
                predictions[m][n] = 0
    list_of_predictions.append(predictions)

    f1 = f1_score(target_set,predictions.round(), labels= labels, average='weighted')
    precision = precision_score(target_set,predictions.round(), labels= labels, average='weighted')
    recall = recall_score(target_set,predictions.round(), labels= labels, average='weighted')
    accuracy = accuracy_score(target_set,predictions.round())
    total_f1 = total_f1 + f1
    total_recall = total_recall +recall
    total_accuracy = total_accuracy + accuracy
    total_precision = total_precision + precision

    statistics_for_batch.append(precision)
    statistics_for_batch.append(recall)
    statistics_for_batch.append(accuracy)
    statistics_for_batch.append(f1)
    statistics.append(statistics_for_batch)
    statistics_for_batch = []
    score_for_autoencoder = []


average_accuracy = total_accuracy/number_of_batches
average_recall = total_recall/number_of_batches
average_precision = total_precision/number_of_batches
average_f1 = total_f1/number_of_batches

output =open('Report_for_cifar_'+str(number_of_classifiers)+'.txt','w')
output.write('\nNumber of classifiers trained '+ str(number_of_classifiers))
output.write('\nTest set split in '+ str(number_of_batches)+' batches')
output.write('\nAverage accuracy achieved on the test set '+ str(average_accuracy))
output.write('\nAverage precision achieved on the test set ' + str(average_precision))
output.write('\nAverage recall achieved on the test set ' + str(average_recall))
output.write('\nAverage F1 achieved on the test set ' + str(average_f1))
output.write('\nStatistics per batch in order Precision, Accuracy,Recall, F1 score :')
for i in range(0,number_of_batches):
    output.write('\nBatch '+str(i)+' Statistics: '+str(statistics[i]))
output.close()

