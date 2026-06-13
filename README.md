# Dry Bean Varieties Classification Project

### **Project Description**

This project implements Machine Learning algorithms to classify different varieties of dry beans using their morphometric characteristics. The objective is to develop a model capable of accurately distinguishing between the distinct varieties, which has important applications in agriculture and quality control.

### **Objectives**

#### **Data Exploration:** Understand the distribution and relationships of the morphometric characteristics of the beans.

#### **Data Preprocessing:** Prepare the dataset through normalization to optimize model performance.

#### **Classification Modeling:** Develop and train a Support Vector Machine (SVM) model for classification.

#### **Model Evaluation:** Analyze model performance using key metrics and visualizations such as learning curves and confusion matrices.

### **Methodology**

#### **1. Data Loading and Exploration**

The dataset used comes from the UCI Machine Learning repository and contains 13,611 instances with 16 morphometric features (such as Area, Perimeter, Major Axis Length, etc.) and a target variable representing 7 different classes of dry beans (SEKER, BARBUNYA, BOMBAY, CALI, SIRA, DERMASON, HOROZ). The absence of null values was confirmed.

#### **2. Data Preprocessing**

Due to the different magnitudes of the input variables, standard scaling (StandardScaler) was applied to the features to ensure no single feature dominated the model training process. The data was split into training and test sets (80% and 20% respectively) using train_test_split with random_state=22 for reproducibility.

#### **3. Modeling (SVM)**

A Support Vector Machine (SVM) model was selected due to its effectiveness in non-linear classification problems and its capability to handle morphometric data. To optimize performance, a hyperparameter search was conducted using GridSearchCV with stratified cross-validation (StratifiedKFold, n_splits=10).

The explored hyperparameters were:

C: [1, 5, 10, 15, 20, 30]

gamma: [0.02, 0.05, 0.08, 0.1, 0.15]

kernel: ['rbf']

#### **4. Model Evaluation**

The optimal model found by GridSearchCV was evaluated using various metrics and visualizations:

Best Accuracy (in cross-validation training): 92.9738%

Best Parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}

Test Set Accuracy: 93.6100%

Learning Curve

The learning curve showed adequate convergence, indicating a good balance between bias and variance, and that the model does not suffer from significant overfitting or underfitting.

Hybrid Confusion Matrix
The confusion matrix provided a detailed view of the model's performance for each class, showing correct and incorrect predictions in both absolute values and proportions. High precision and recall were observed across most classes.

### **Classification Report**

The classification report detailed the precision, recall, and f1-score for each of the 7 classes, as well as the support. The metrics were consistently high, exceeding 90% for most classes, which validates the model's robustness.

### **Conclusion**

The optimized SVM model proved to be highly effective in classifying dry bean varieties, achieving an accuracy above 93% on the test set. The evaluation results, including the learning curve, confusion matrix, and classification report, confirm the robustness and reliability of the model.

This project establishes a solid foundation for future research, which could include the exploration of other algorithms, advanced feature engineering, or the implementation of deep learning techniques to compare and improve performance.
