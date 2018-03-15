
import pandas as pd
import numpy as np
import random



data = pd.read_csv("jester-data-1.csv")
d = data.to_latex()
text_file = open("Output.txt", "w")
text_file.write(d)
text_file.close()
data.drop(data.columns[0], axis=1, inplace = True)
zero = np.zeros((data.shape[0],data.shape[1]))
unknown_np = zero + 99

data_numpy=data.as_matrix()
for i in range(0,data_numpy.shape[0]):
    for j in range(0,data_numpy.shape[1]):
        if random.random() >= 0.9:
            x = data_numpy[i][j]
            data_numpy[i][j] = unknown_np[i][j]
            unknown_np[i][j] = x


n_features = 2

user_ratings = data_numpy
latent_user_preferences = np.random.random((user_ratings.shape[0], n_features))
latent_item_features = np.random.random((user_ratings.shape[1],n_features))


def predict_rating(user_id,item_id):
    """ Predict a rating given a user_id and an item_id.
    """
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_item_features[item_id]
    return user_preference.dot(item_preference)


def train(user_id, item_id, rating, alpha=0.0005):
    prediction_rating = predict_rating(user_id, item_id)
    err = (prediction_rating - rating)
    latent_user_preferences[user_id] -= alpha * err * latent_item_features[item_id]
    latent_item_features[item_id] -= alpha * err * latent_user_preferences[user_id]
    return err


def sgd(iterations=30):
    """ Iterate over all users and all items and train for
        a certain number of iterations
    """
    for i in range(0, iterations):
        error = []
        for user_id in range(0, latent_user_preferences.shape[0]):
            for item_id in range(0, latent_item_features.shape[0]):
                rating = user_ratings[user_id][item_id]
                if not np.isnan(rating) and rating !=99:
                    err = train(user_id, item_id, rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()
        mse = np.sqrt(mse)
        if i%5 == 0:
            print(mse)

sgd()

predictions = latent_user_preferences.dot(latent_item_features.T)
values = [zip(user_ratings[i], predictions[i]) for i in range(0,predictions.shape[0])]
comparison_data = pd.DataFrame(values)
comparison_data.columns = data.columns
#comparison_data.applymap(lambda (x,y): "(%2.3f|%2.3f)"%(x,y))
d = comparison_data.to_latex()
text_file = open("comparison.txt", "w")
text_file.write(d)
text_file.close()


count = 0

for i in range(0, len(unknown_np)):
    for j in range(0,len(unknown_np[0])):
        if unknown_np[i][j] != 99:
            error_sample = unknown_np[i][j] - predictions[i][j]
            square_error = error_sample * error_sample
            count = count +1


final_error = np.sqrt(square_error/count)
print('MSE on the test set = ' + str(final_error))
