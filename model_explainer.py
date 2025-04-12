class ModelExplainer:
    """Class to provide explanations for different machine learning models"""
    
    def get_explanations(self):
        """Return explanations for all models"""
        return {
            "Logistic Regression": self.explain_logistic_regression(),
            "K-Nearest Neighbors": self.explain_knn(),
            "Decision Tree": self.explain_decision_tree(),
            "Random Forest": self.explain_random_forest(),
            "Support Vector Machine": self.explain_svm(),
            "Naive Bayes": self.explain_naive_bayes()
        }
    
    def explain_logistic_regression(self):
        """Explain Logistic Regression model"""
        return {
            "description": """
                <p>Logistic Regression is a statistical model that uses a logistic function to model a binary 
                dependent variable. Despite its name, it's a classification algorithm, not regression.</p>
                
                <p>It works by applying a logistic function to a linear combination of features to estimate 
                the probability of a certain event (in this case, survival). The model calculates the 
                probability that a passenger survived based on their attributes and classifies them as 
                survivors if this probability exceeds 0.5.</p>
                
                <p>The model learns weights (coefficients) for each feature, which indicate how strongly each 
                feature influences the prediction. These weights can provide insights into which features are 
                most important for predicting survival.</p>
            """,
            "pros": [
                "Simple and easy to interpret",
                "Provides probability estimates",
                "Works well with linearly separable data",
                "Less prone to overfitting with small datasets",
                "Feature coefficients provide insights into importance"
            ],
            "cons": [
                "Assumes linear relationship between features and log-odds",
                "May not capture complex relationships in data",
                "Can struggle with highly correlated features",
                "Requires feature scaling for optimal performance"
            ]
        }
    
    def explain_knn(self):
        """Explain K-Nearest Neighbors model"""
        return {
            "description": """
                <p>K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm. It doesn't 
                learn a model, but instead memorizes the training data.</p>
                
                <p>When making a prediction, KNN finds the K training examples that are closest to the new 
                data point in feature space. It then classifies the new point by taking a majority vote of 
                its neighbors' classes. In our case, it would predict survival if the majority of the K 
                nearest passengers in the training data survived.</p>
                
                <p>The "closeness" is typically measured using Euclidean distance, though other distance 
                metrics can be used. The value of K is a hyperparameter that needs to be tuned.</p>
            """,
            "pros": [
                "Simple and intuitive algorithm",
                "No training phase - predictions made directly from data",
                "Can capture complex decision boundaries",
                "Can be effective with sufficient, representative data",
                "No assumptions about data distribution"
            ],
            "cons": [
                "Computationally expensive for large datasets",
                "Sensitive to irrelevant features",
                "Requires feature scaling",
                "The optimal value of K can be difficult to determine",
                "Performance degrades with high-dimensional data"
            ]
        }
    
    def explain_decision_tree(self):
        """Explain Decision Tree model"""
        return {
            "description": """
                <p>A Decision Tree is a flowchart-like model that makes decisions based on asking a series of 
                questions about the features. Each internal node represents a 'test' on an attribute, each 
                branch represents the outcome of the test, and each leaf node represents a prediction.</p>
                
                <p>The algorithm works by recursively splitting the data based on feature values to create 
                the purest possible child nodes. The 'purity' is measured using metrics like Gini impurity or 
                entropy. For the Titanic dataset, the tree might first split on 'Sex', then on 'Age' for males, 
                and on 'Class' for females.</p>
                
                <p>Decision trees are highly interpretable as they can be visualized and the decision path 
                can be followed from root to leaf.</p>
            """,
            "pros": [
                "Highly interpretable - decision process can be visualized",
                "Handles both numerical and categorical data",
                "Requires minimal data preprocessing",
                "Automatically handles feature interactions",
                "Can handle missing values"
            ],
            "cons": [
                "Prone to overfitting, especially with deep trees",
                "High variance - small changes in data can result in very different trees",
                "Can create biased trees if classes are imbalanced",
                "May not generalize well to unseen data",
                "Struggles with diagonal decision boundaries"
            ]
        }
    
    def explain_random_forest(self):
        """Explain Random Forest model"""
        return {
            "description": """
                <p>Random Forest is an ensemble learning method that constructs multiple decision trees during 
                training and outputs the class that is the mode of the classes from individual trees.</p>
                
                <p>Each tree is built using a subset of the training data (bootstrap sample) and a random 
                subset of features at each split. This randomness helps to make the trees less correlated and 
                improves the model's generalization ability.</p>
                
                <p>For the Titanic dataset, Random Forest combines the predictions from many decision trees, 
                each considering different combinations of passenger attributes. This approach often captures 
                more patterns in the data than a single decision tree.</p>
            """,
            "pros": [
                "Higher accuracy than single decision trees",
                "Less prone to overfitting",
                "Provides feature importance measures",
                "Handles large datasets with higher dimensionality",
                "Robust to outliers and non-linear data"
            ],
            "cons": [
                "Less interpretable than a single decision tree",
                "Computationally more intensive than simpler models",
                "Can still overfit in certain scenarios",
                "May not perform well if trees are correlated",
                "Requires more hyperparameter tuning"
            ]
        }
    
    def explain_svm(self):
        """Explain Support Vector Machine model"""
        return {
            "description": """
                <p>Support Vector Machine (SVM) is a powerful classification algorithm that finds the optimal 
                hyperplane that maximizes the margin between different classes.</p>
                
                <p>In the case of the Titanic dataset, SVM tries to find the best boundary that separates 
                survivors from non-survivors in the feature space. It focuses on the 'support vectors' - the 
                data points that are closest to the decision boundary and influence its position.</p>
                
                <p>SVM can also handle non-linearly separable data by using kernel functions that transform 
                the data into a higher-dimensional space where a linear separator might exist. Common kernels 
                include linear, polynomial, and radial basis function (RBF).</p>
            """,
            "pros": [
                "Effective in high-dimensional spaces",
                "Memory efficient as it uses only a subset of training points",
                "Versatile through different kernel functions",
                "Good generalization for both linear and non-linear problems",
                "Robust against overfitting"
            ],
            "cons": [
                "Not suitable for large datasets due to computational complexity",
                "Selecting the appropriate kernel can be challenging",
                "Difficult to interpret the model's decisions",
                "Sensitive to feature scaling",
                "Does not directly provide probability estimates"
            ]
        }
    
    def explain_naive_bayes(self):
        """Explain Naive Bayes model"""
        return {
            "description": """
                <p>Naive Bayes is a probabilistic classifier based on Bayes' theorem with the 'naive' assumption 
                of conditional independence between every pair of features given the class.</p>
                
                <p>Despite this simplifying assumption, Naive Bayes performs surprisingly well in many 
                real-world situations. For the Titanic dataset, it calculates the probability of survival given 
                a passenger's attributes by applying Bayes' theorem.</p>
                
                <p>The classifier first calculates the prior probability of each class (survived vs. not survived) 
                and the conditional probability of each feature given each class from the training data. When 
                predicting, it applies Bayes' theorem to find the posterior probability of each class given 
                the features.</p>
                
                <p>There are several variants of Naive Bayes; for continuous features like Age and Fare, 
                Gaussian Naive Bayes is commonly used, which assumes features follow a normal distribution.</p>
            """,
            "pros": [
                "Simple, fast, and efficient",
                "Works well with small datasets",
                "Performs well even with the independence assumption",
                "Handles missing values well",
                "Not sensitive to irrelevant features",
                "Provides probability estimates"
            ],
            "cons": [
                "Assumes feature independence, which is rarely true in practice",
                "Can be outperformed by more sophisticated models",
                "May give poor estimates of probabilities",
                "Sensitive to how input data is prepared",
                "Known as the 'zero frequency problem': if a categorical variable has a category in test data that wasn't observed in training data, the model will assign a probability of zero"
            ]
        }
