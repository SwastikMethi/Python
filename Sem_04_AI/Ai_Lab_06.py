import numpy as np
import pandas as pd
from math import log2

##Q-1
#(a)
data = {
    'High School GPA': [3.8, 3.2, 4, 2.5, 6],
    'SAT Score': [1450, 1300, 1500, 1200, 1400],
    'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes','No'],
    'Recommendation Letter Strength': ['Strong', 'Weak', 'Strong', 'Weak', 'Strong'],
    'Admission Status': ['Adimitted', 'Not Admitted', 'Adimitted', 'Not Admitted', 'Adimitted']
}

df = pd.DataFrame(data)

class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.children = {}

def entropy(labels):
    """Calculate the entropy of a list of labels."""
    label_counts = pd.Series(labels).value_counts()
    total_samples = len(labels)
    entropy_value = 0
    for count in label_counts:
        probability = count / total_samples
        entropy_value -= probability * log2(probability)
    return entropy_value

def information_gain(data, attribute):
    """Calculate the information gain for a given attribute."""
    total_samples = len(data)
    attribute_values = data[attribute].unique()
    attribute_entropy = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        subset_entropy = entropy(subset['Admission Status'])
        subset_weight = len(subset) / total_samples
        attribute_entropy += subset_weight * subset_entropy
    gain = entropy(data['Admission Status']) - attribute_entropy
    return gain

def select_best_attribute(data):
    """Select the best attribute to split on based on maximum information gain."""
    attributes = list(data.columns[:-1])
    best_attribute = None
    best_gain = -1
    for attribute in attributes:
        gain = information_gain(data, attribute)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    return best_attribute

def build_tree(data):
    """Recursively build the decision tree."""
    if len(data['Admission Status'].unique()) == 1:
        label = data['Admission Status'].iloc[0]
        return Node(label=label)
    
    if len(data.columns) == 1:
        label = data['Admission Status'].mode()[0]
        return Node(label=label)
    
    best_attribute = select_best_attribute(data)
    
    node = Node(attribute=best_attribute)
    
    attribute_values = data[best_attribute].unique()
    for value in attribute_values:
        subset = data[data[best_attribute] == value]
        if len(subset) == 0:
            label = data['Admission Status'].mode()[0]
            node.children[value] = Node(label=label)
        else:
            node.children[value] = build_tree(subset.drop(columns=[best_attribute]))
    
    return node

def print_tree(node, depth=0):
    """Print the decision tree."""
    if node.label is not None:
        print('  ' * depth + 'Predict:', node.label)
    else:
        print('  ' * depth + 'Attribute:', node.attribute)
        for value, child_node in node.children.items():
            print('  ' * (depth + 1) + f'Value: {value}')
            print_tree(child_node, depth + 2)

tree = build_tree(df)

print("Decision Tree:")
print_tree(tree)

#(b)
def evaluate_tree(node, sample):
    """Traverse the decision tree to predict the label for a given sample."""
    if node.label is not None:
        return node.label
    
    attribute_value = sample[node.attribute]
    if attribute_value not in node.children:
        return max(node.children, key=lambda k: node.children[k].label)
    else:
        next_node = node.children[attribute_value]
        return evaluate_tree(next_node, sample)

test_data = {
    'High School GPA': 3.5,
    'SAT Score': 1350,
    'Extracurricular Activities': 'Yes',
    'Recommendation Letter Strength': 'Weak'
}
prediction = evaluate_tree(tree, test_data)
print("Prediction for the test data:", prediction)


## Q-2
import numpy as np
import pandas as pd

# Define the dataset
data = {
    'High School GPA': [3.8, 3.2, 4, 2.5, 6],
    'SAT Score': [1450, 1300, 1500, 1200, 1400],
    'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Recommendation Letter Strength': ['Strong', 'Weak', 'Strong', 'Weak', 'Strong'],
    'Admission Status': ['Admitted', 'Not Admitted', 'Admitted', 'Not Admitted', 'Admitted']
}

df = pd.DataFrame(data)

class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute  # Attribute to split on
        self.value = value          # Value of the attribute to split on
        self.result = result        # Result at this node (if leaf node)
        self.children = {}          # Children of this node (dictionary of value: node)

def gini_impurity(labels):
    """Calculate the Gini impurity for a list of labels."""
    n = len(labels)
    if n == 0:
        return 0
    
    # Count the occurrences of each class
    counts = np.unique(labels, return_counts=True)[1]
    
    # Calculate the Gini impurity
    impurity = 1 - sum((count / n) ** 2 for count in counts)
    return impurity

def information_gain(parent, splits):
    """Calculate the information gain."""
    total_size = sum(len(split) for split in splits)
    gain = gini_impurity(parent) - sum((len(split) / total_size) * gini_impurity(split['Admission Status']) for split in splits)
    return gain

def split_data(data, attribute):
    """Split the dataset based on the values of a given attribute."""
    splits = {}
    for index, row in data.iterrows():
        value = row[attribute]
        if value not in splits:
            splits[value] = {'High School GPA': [], 'SAT Score': [], 'Extracurricular Activities': [], 'Recommendation Letter Strength': [], 'Admission Status': []}
        for col in data.columns:
            splits[value][col].append(row[col])
    return splits

def build_tree(data):
    """Recursively build the decision tree."""
    if len(data['Admission Status'].unique()) == 1:
        label = data['Admission Status'].iloc[0]
        return Node(result=label)
    
    if len(data.columns) == 1:
        label = data['Admission Status'].mode()[0]
        return Node(result=label)
    
    attributes = data.columns[:-1]
    best_attribute = None
    best_gain = -1
    for attribute in attributes:
        splits = split_data(data, attribute)
        gain = information_gain(data, splits.values())
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    splits = split_data(data, best_attribute)
    
    node = Node(attribute=best_attribute)
    
    for value, split_data in splits.items():
        node.children[value] = build_tree(pd.DataFrame(split_data))
    
    return node

def print_tree(node, depth=0):
    """Print the decision tree."""
    if node.result is not None:
        print('  ' * depth + 'Predict:', node.result)
    else:
        print('  ' * depth + 'Attribute:', node.attribute)
        for value, child_node in node.children.items():
            print('  ' * (depth + 1) + f'Value: {value}')
            print_tree(child_node, depth + 2)

tree = build_tree(df)

print("Decision Tree:")
print_tree(tree)

