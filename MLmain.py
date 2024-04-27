from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


print(iris_data.head())


# Define features and target
X = iris_data[iris.feature_names]  # Features
y = iris_data['species']           # Target

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
tree_model = DecisionTreeClassifier(random_state=42)

# teach the Tree using our training team clue and treasures

tree_model.fit(X_train, y_train)

#let the tree predict the treasures for the testing team

predictions = tree_model.predict(X_test)

# importing a tool to see how good the tree did

from sklearn.metrics import accuracy_score

# check the accuracy of our tree

accuracy = accuracy_score(y_test, predictions)
print("Accuracy of our Decission Tree: ", accuracy)