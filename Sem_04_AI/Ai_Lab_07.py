# import numpy as np

# data = np.array([
#     [1, 2, 1],
#     [2, 3, -1],
#     [3, 2, 1],
#     [4, 3, -1],
#     [2, 1, 1]
# ])


# weights = np.ones(len(data)) / len(data)

# def weak_classifier(data, feature_index, threshold, direction):
    
#     predictions = np.ones(len(data))
#     if direction == 'left':
#         predictions[data[:, feature_index] <= threshold] = -1
#     else:
#         predictions[data[:, feature_index] > threshold] = -1
#     return predictions

# def calculate_error(predictions, labels, weights):
    
#     error = np.sum(weights * (predictions != labels))
#     return error

# def update_weights(weights, error,labels):
    
#     beta = error / (1 - error)
#     weights *= np.power(beta, 1 - (predictions == labels))


# feature_index = 0
# threshold = 2.5
# direction = 'left'
# predictions = weak_classifier(data, feature_index, threshold, direction)


# error = calculate_error(predictions, data[:, -1], weights)
# print(f"Iteration 1 - Error: {error}")


# update_weights(weights, error,data[:,-1])


# alpha = 0.5 * np.log((1 - error) / error)
# print(f"Iteration 1 - Classifier Weight (alpha): {alpha}")


# weights /= np.sum(weights)


# feature_index = 1
# threshold = 2.5
# direction = 'right'
# predictions = weak_classifier(data, feature_index, threshold, direction)


# error = calculate_error(predictions, data[:, -1], weights)
# print(f"Iteration 2 - Error: {error}")


# update_weights(weights, error,data[:, -1])


# alpha = 0.5 * np.log((1 - error) / error)
# print(f"Iteration 2 - Classifier Weight (alpha): {alpha}")


# weights /= np.sum(weights)


# def final_hypothesis(data, classifiers, alphas):
   
#     final_predictions = np.zeros(len(data))
#     for i in range(len(classifiers)):
#         final_predictions += alphas[i] * weak_classifier(data, classifiers[i][0], classifiers[i][1], classifiers[i][2])

   
#     new_point = np.array([3, 3])
#     final_prediction = np.sign(np.sum(alphas[i] * weak_classifier(new_point.reshape(1, -1), classifiers[i][0], classifiers[i][1], classifiers[i][2]) for i in range(len(classifiers))))
#     return final_prediction


# classifiers = [(0, 2.5, 'left'), (1, 2.5, 'right')]
# alphas = [0.682, 0.881] 
# prediction = final_hypothesis(np.array([[3, 3]]), classifiers, alphas)
# print(f"Final Hypothesis Prediction for (3, 3): {prediction}")

# import numpy as np
# from sklearn.tree import DecisionTreeRegressor

# y = np.array([2.5, 0.5, 3.5, 1.5, 2.0])

# initial_prediction = np.mean(y)
# print("Initial prediction:", initial_prediction)

# residuals_1 = y - initial_prediction

# tree_1 = DecisionTreeRegressor(max_depth=1)
# tree_1.fit(X.reshape(-1, 1), residuals_1)

# learning_rate = 0.1
# updated_prediction_1 = initial_prediction + learning_rate * tree_1.predict(X.reshape(-1, 1))
# print("Updated prediction after iteration 1:", updated_prediction_1)

# # Residuals for the second iteration
# residuals_2 = y - updated_prediction_1

# # Fit a decision tree to new residuals with depth 1
# tree_2 = DecisionTreeRegressor(max_depth=1)
# tree_2.fit(X.reshape(-1, 1), residuals_2)

# # Update the model
# updated_prediction_2 = updated_prediction_1 + learning_rate * tree_2.predict(X.reshape(-1, 1))
# print("Updated prediction after iteration 2:", updated_prediction_2)

# # Predicted value for feature x = 3.5
# new_feature = 3.5
# predicted_value = updated_prediction_2[np.where(x == new_feature)]
# print("Predicted value for feature x = 3.5 after two iterations:", predicted_value)

