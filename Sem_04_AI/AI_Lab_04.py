import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;

data=pd.read_csv("G:\Course notes\divorce_data.csv",delimiter=";")
data['intercept'] = 1
print(data.head())

X = data.drop('Q1', axis=1)
y = data['Q1']
X_shape = X.shape
X_type  = type(X)
y_shape = y.shape
y_type  = type(y)
print(f'X: Type-{X_type}, Shape-{X_shape}')
print(f'y: Type-{y_type}, Shape-{y_shape}')

data.isnull().sum()

scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test : {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test : {y_test.shape}')

X_train_shape = X.shape
y_train_shape = y.shape
X_test_shape  = X.shape
y_test_shape  = y.shape

print(f"X_train: {X_train_shape} , y_train: {y_train_shape}")
print(f"X_test: {X_test_shape} , y_test: {y_test_shape}")
assert (X_train.shape[0]==y_train.shape[0] and X_test.shape[0]==y_test.shape[0]), "Check your splitting carefully"

class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.cross_entropy_error = []
        self.eps = 1e-7

    def sigmoid(self, z):
        sig_z = 1/(1+np.exp(-z))
        assert (z.shape==sig_z.shape), 'Error in sigmoid implementation. Check carefully'
        return sig_z
    
    def cross_entropy(self, y_true, y_pred):
        cross_entropy =  -np.mean(y_true*np.log(y_pred+self.eps) + (1-y_true)*np.log(1-y_pred+self.eps))
        return cross_entropy
    
    def fit(self, X, y):
        num_examples = X.shape[0]
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        for i in range(self.max_iterations):
            z =  np.dot(X, self.weights)
            y_pred =  self.sigmoid(z)
            gradient = np.mean((y-y_pred)*X.T, axis=1)
            self.weights =  self.weights + self.learning_rate*gradient
            cross_entropy =  self.cross_entropy(y, y_pred)
            self.cross_entropy_error.append(cross_entropy)
    
    def predict_proba(self, X):
        if self.weights is None:
            raise Exception("Fit the model before prediction")
        z =  np.dot(X, self.weights)
        probabilities =  self.sigmoid(z)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        binary_predictions = np.array(list(map(lambda x: 1 if x>threshold else 0, self.predict_proba(X))))
        return binary_predictions

model =  MyLogisticRegression(learning_rate=0.1, max_iterations=5000)
train_cross_entropy = model.cross_entropy(y_train, model.predict_proba(X_train))
print("Cross entropy cost on training data:", train_cross_entropy)

test_cross_entropy = model.cross_entropy(y_test, model.predict_proba(X_test))
print("Cross entropy cost on testing data:", test_cross_entropy)
plt.plot([i+1 for i in range(len(model.cross_entropy_error))], model.cross_entropy_error)
plt.title("Cross entropy error curve")
plt.xlabel("Iteration num")
plt.ylabel("Cross entropy (-ve log-likelihood)")
plt.show()
y_pred = model.predict(X_test)

def accuracy(y_true,y_pred):
    '''Compute accuracy.
    Accuracy = (Correct prediction / number of samples)
    Args:
        y_true : Truth binary values (num_examples, )
        y_pred : Predicted binary values (num_examples, )
    Returns:
        accuracy: scalar value
    '''

    
    accuracy =  np.mean(y_true==y_pred)
    return accuracy

print(f"Accuracy on training data: {accuracy(y_train, model.predict(X_train))}")
print(f"Accuracy on testing data: {accuracy(y_test, model.predict(X_test))}")

