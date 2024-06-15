import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read data from CSV using pandas
def read_data(file):
    return pd.read_csv(file)

# Convert categorical columns
def categorical_column_transformer(data):
    label_encoder_gender = LabelEncoder()
    label_encoder_city = LabelEncoder()

    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
    data['City'] = label_encoder_city.fit_transform(data['City'])

    return data, label_encoder_gender, label_encoder_city

# Normalize numerical columns
def normalize(data):
    numerical_columns = ['Age', 'Income']
    sc = StandardScaler()

    data[numerical_columns] = sc.fit_transform(data[numerical_columns])
    return data, sc

# Calculate KNN accuracy
def knn_accuracy(data, k_values):
    X = data[['Age', 'Income', 'Gender', 'City']]
    y = data['Gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Print accuracy, precision, recall, and F1 values
        with np.errstate(divide='ignore', invalid='ignore'):
            print(f"K={k}, Accuracy: {accuracy_score(y_test, y_pred)}, Precision: {precision_score(y_test, y_pred, zero_division=1.0)}, Recall: {recall_score(y_test, y_pred)}, F1: {f1_score(y_test, y_pred)}")

# Calculate MLP accuracy
def mlp_accuracy(data, layer_sizes):
    X = data[['Age', 'Income', 'Gender', 'City']]
    y = data['Gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for size in layer_sizes:
        mlp = MLPClassifier(hidden_layer_sizes=size, max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        # Print accuracy, precision, recall, and F1 values
        with np.errstate(divide='ignore', invalid='ignore'):
            print(f"Hidden Layer Size: {size}, Accuracy: {accuracy_score(y_test, y_pred)}, Precision: {precision_score(y_test, y_pred, zero_division=1.0)}, Recall: {recall_score(y_test, y_pred)}, F1: {f1_score(y_test, y_pred)}")

# Calculate Naive Bayes accuracy
def nb_accuracy(data):
    X = data[['Age', 'Income', 'Gender', 'City']]
    y = data['Gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    # Print accuracy, precision, recall, and F1 values
    print(f"NB Accuracy: {accuracy_score(y_test, y_pred)}, Precision: {precision_score(y_test, y_pred)}, Recall: {recall_score(y_test, y_pred)}, F1: {f1_score(y_test, y_pred)}")

# Main function
def main():
    file = 'data.csv'

    # Read data
    data = read_data(file)
    print("Read data:")
    print(data)
    print("\n")

    # Transform categorical columns
    data, label_encoder_gender, label_encoder_city = categorical_column_transformer(data)
    print("Transformed Data:")
    print(data)

    # Print the classes of label encoders
    print("Gender Classes: ", label_encoder_gender.classes_)
    print("City Classes: ", label_encoder_city.classes_)
    print("\n")

    # Normalize numerical columns
    data, sc = normalize(data)
    print("Normalized Data:")
    print(data)
    print("\n")

    # Calculate and print KNN values
    print("KNN Values:")
    k_values = [3, 7, 11]
    knn_accuracy(data, k_values)
    print("\n")

    # Calculate and print MLP values
    print("MLP Values:")
    layer_sizes = [32, (32, 32), (32, 32, 32)]
    mlp_accuracy(data, layer_sizes)
    print("\n")

    # Calculate and print NB values
    print("NB Values:")
    nb_accuracy(data)

if __name__ == "__main__":
    main()
