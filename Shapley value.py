#!/usr/bin/env python
# coding: utf-8

# In[1]:



pip install shap scikit-learn pyGWO

Step-by-Step Code

# Import necessary libraries
import shap
import numpy as np
import pandas as pd
from sklearn.datasets import load_dataset_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pyGWO import GWO

# Load and preprocess the dataset
def load_data():
    # Load the dataset cancer dataset from sklearn
    data = load_dataset_cancer()
    X = data.data
    y = data.target
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Step 1: Use Kernel Shapley Value to compute feature importance
def compute_kernel_shapley(X_train, y_train):
    model = SVC(kernel="linear", probability=True)  # Using SVM as the classifier
    model.fit(X_train, y_train)
    
    # Use shap's KernelExplainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
    shap_values = explainer.shap_values(X_train, nsamples=100)  # nsamples controls sampling; adjust for efficiency

    # Calculate the mean absolute Shapley values for each feature
    feature_importance = np.abs(shap_values[1]).mean(axis=0)
    return feature_importance

# Step 2: Define Improved Gray Wolf Optimizer (IGWO)
# Here is a simplified version of GWO. You can modify it to implement IGWO or use a library for GWO.
class IGWO(GWO):
    def __init__(self, nwolves=5, max_iter=20):
        super().__init__(nwolves, max_iter)
        
    def optimize(self, cost_function, dim):
        """
        Optimize the cost function.
        - cost_function: A function to minimize, receives a binary vector indicating selected features.
        - dim: Dimension of the search space, number of features in this case.
        """
        # Call base optimizer, you may extend this to improve IGWO (e.g., adaptive parameters)
        return super().optimize(cost_function, dim)

# Step 3: Define Fitness Function for IGWO
def fitness_function(feature_indices):
    # Select features based on indices given by IGWO
    selected_features = np.where(feature_indices > 0.5, 1, 0)  # Binary decision rule for feature inclusion
    if selected_features.sum() == 0:
        return 1  # If no features are selected, return worst score
    
    # Apply feature selection and evaluate with cross-validation on SVM
    X_selected = X_train[:, selected_features == 1]
    model = SVC(kernel="linear", probability=True)
    scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='accuracy')
    return 1 - scores.mean()  # Minimize the complement of accuracy

# Step 4: Use IGWO for Feature Selection Optimization
def run_feature_selection(X_train, y_train, feature_importance):
    # Initialize IGWO
    nwolves = 10
    max_iter = 20
    gwo = IGWO(nwolves=nwolves, max_iter=max_iter)

    # Sort features by importance and prioritize in IGWO
    sorted_features = np.argsort(-feature_importance)
    
    # Run GWO optimization
    optimal_features = gwo.optimize(lambda x: fitness_function(x[sorted_features]), len(feature_importance))

    # Get final selected feature indices
    selected_feature_indices = np.where(optimal_features > 0.5)[0]
    return selected_feature_indices

# Step 5: Evaluate the selected features on test data using SVM
def evaluate_model(X_train, y_train, X_test, y_test, selected_features):
    # Train SVM on selected features
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train[:, selected_features], y_train)
    
    # Test on the test set
    y_pred = model.predict(X_test[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main Execution
if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 1: Compute feature importance using Kernel Shapley
    feature_importance = compute_kernel_shapley(X_train, y_train)
    
    # Step 2: Run IGWO for feature selection based on Shapley values
    selected_features = run_feature_selection(X_train, y_train, feature_importance)
    
    # Step 3: Evaluate the model performance on the selected features
    accuracy = evaluate_model(X_train, y_train, X_test, y_test, selected_features)
    
    print("Selected Feature Indices:", selected_features)
    print("Test Accuracy with Selected Features:", accuracy)


# In[ ]:




