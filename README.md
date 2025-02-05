# Support Vector Machine (SVM) Classifier for MNIST Dataset

## Overview

This project implements a machine learning pipeline using a Support Vector Machine (SVM) model to classify digits from the MNIST dataset. The dataset consists of handwritten digits (0-9), and the goal is to train an SVM model to accurately predict digit classes.

## Features

- Load MNIST dataset using Pandas.

- Perform exploratory data analysis (EDA).

- Preprocess data by scaling features using StandardScaler.

- Split dataset into training and test sets.

- Train a Support Vector Machine model using Scikit-Learn.

- Evaluate model performance using classification metrics and confusion matrix visualization.


## Dataset

The MNIST dataset used in this project is loaded from a CSV file hosted online. The dataset contains 4,000 samples with 786 columns:

- id: Unique identifier for each sample.

- class: Target variable (digit 0-9).

- pixel1 - pixel784: Pixel intensity values (features).

## Installation & Setup

1. Clone this repository:
git clone https://github.com/Phionanamugga/svc_ml_model 
cd your-repo 

2. Install required dependencies: 
pip install pandas scikit-learn matplotlib seaborn 

## Usage
Run the Python script to train and evaluate the SVM model: 
python svm_classifier.py 

## Code Breakdown
1. Importing Libraries 
- import pandas as pd
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- from sklearn.svm import SVC
- from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
- import matplotlib.pyplot as plt
- import seaborn as sns 

2. Load Dataset 
df = pd.read_csv('https://raw.githubusercontent.com/Phionanamugga/teaching/refs/heads/main/datasets/mnist.csv') 

3. Exploratory Data Analysis (EDA)
print(df.head())
print(df.info())
print(df.describe())

4. Data Preprocessing
X = df.drop('class', axis=1)
y = df['class']
scaler = StandardScaler()
X = scaler.fit_transform(X)

5. Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

6. Training the Model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

7. Evaluating the Model
y_pred = svm_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

8. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

## Results

The trained SVM model achieved an accuracy of 91.63% on the test dataset. The confusion matrix visualization provides insights into model performance across different digit classes.

## License
This project is licensed under the MIT License.
MIT License

Copyright (c) 2025 Phiona Namugga

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contributing

If you’d like to contribute to this project, feel free to submit a pull request or open an issue.


## Contact

For any questions or suggestions, please reach out to me via GitHub.