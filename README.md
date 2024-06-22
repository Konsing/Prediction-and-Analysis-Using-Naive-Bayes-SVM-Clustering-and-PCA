# Productivity Prediction and Analysis Using Naive Bayes, SVM, Clustering, and PCA

## Project Overview
This Jupyter notebook demonstrates the application of various machine learning techniques, including Naive Bayes classification, Support Vector Machine (SVM) modeling, clustering, and Principal Component Analysis (PCA), to predict and analyze the productivity of garment employees. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Files in the Repository
- **Naive_Bayes_SVM_Clustering_PCA_Analysis.ipynb**: This Jupyter notebook contains the code and explanations for the various tasks performed in the project.
- **bitstrings.csv**: Dataset used for PCA and clustering analysis.
- **garments_worker_productivity.csv**: Dataset used for productivity prediction and analysis.

## How to Use
1. **Prerequisites**:
   - Python 3.x
   - Jupyter Notebook or JupyterLab
   - Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

2. **Installation**:
   Ensure you have the required packages installed. You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Running the Notebook**:
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook Naive_Bayes_SVM_Clustering_PCA_Analysis.ipynb
     ```
   - Execute the cells in the notebook sequentially to perform the various tasks and analyses.

## Sections in the Notebook

### 1. Introduction
This section introduces the project, outlining the datasets used (Garment Worker Productivity and Bitstrings) and the key tasks to be performed, including Naive Bayes classification, SVM modeling, clustering, and PCA.

### 2. Data Preprocessing
#### Description:
Prepare the Garment Worker Productivity dataset for model training.
#### Key Steps:
   - Load the dataset.
   - Handle missing values and encode categorical variables.
   - Normalize numerical features for better training performance.

### 3. Naive Bayes Classification
#### Description:
Build and evaluate a Naive Bayes classifier to predict the productivity of garment workers.
#### Key Steps:
   - Split the dataset into training and testing sets.
   - Train the Naive Bayes classifier.
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Visualize the results using a confusion matrix.

### 4. SVM Model
#### Description:
Develop and evaluate an SVM model for predicting garment worker productivity.
#### Key Steps:
   - Train the SVM model with the training dataset.
   - Evaluate the model using accuracy, precision, recall, and F1-score.
   - Visualize the results using a confusion matrix.

### 5. Clustering Analysis
#### Description:
Apply k-means and agglomerative clustering to analyze the bitstrings dataset.
#### Key Steps:
   - Perform k-means clustering with specified cluster numbers.
   - Perform agglomerative clustering and compare with k-means.
   - Visualize the frequency of each cluster.
   - Discuss the differences between k-means and agglomerative clustering.

### 6. Principal Component Analysis (PCA)
#### Description:
Use PCA for feature extraction and visualization.
#### Key Steps:
   - Perform k-means clustering with k=2.
   - Apply PCA to reduce dimensions to 2 components.
   - Visualize the projected data points using a scatter plot.
   - Identify and report the feature with the highest positive weight in the first principal component.

## Visualization
The notebook includes various visualizations to support the analysis, such as confusion matrices, cluster frequency charts, and scatter plots of PCA results. Each section's visualizations help in understanding the data and the results of the applied techniques.

## Conclusion
This notebook provides a comprehensive approach to productivity prediction and analysis using Naive Bayes, SVM, clustering, and PCA. By following the steps in the notebook, users can replicate the analyses on similar datasets or extend them to other data.

If you have any questions or encounter any issues, please feel free to reach out for further assistance.