import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Customer ID': [1, 2, 3, 4, 5],
    'Age': [25, 34, 40, 22, 35],
    'Income': ['High', 'Low', 'Medium', 'Low', 'High'],
    'Feedback Word Count': [45, 30, 25, 20, 50],
    'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

df = pd.DataFrame(data)

df['Income'] = df['Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Sentiment'] = df['Sentiment'].map({'Negative': 0, 'Positive': 1})

X = df[['Age', 'Income', 'Feedback Word Count']]
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

new_feedback = pd.DataFrame({'Age': [30], 'Income': ['Medium'], 'Feedback Word Count': [40]})
new_feedback['Income'] = new_feedback['Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
new_prediction = nb_classifier.predict(new_feedback)

print("Predicted Sentiment for the new feedback:", "Positive" if new_prediction[0] == 1 else "Negative")

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

