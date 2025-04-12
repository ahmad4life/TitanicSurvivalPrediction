import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

class ModelTrainer:
    def __init__(self, X_train, y_train, feature_names):
        """Initialize the ModelTrainer with training data and feature names"""
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        
        # Initialize models
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB()
        }
        
        # Dictionary to track trained models
        self.trained_models = {model_name: None for model_name in self.models.keys()}
        
        # Define simplified hyperparameter grids for faster training
        self.param_grids = {
            "Logistic Regression": {
                'C': [0.1, 1],
                'solver': ['liblinear']
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5],
                'weights': ['uniform']
            },
            "Decision Tree": {
                'max_depth': [5, 10],
                'criterion': ['gini']
            },
            "Random Forest": {
                'n_estimators': [50],
                'max_depth': [10]
            },
            "SVM": {
                'C': [1],
                'kernel': ['linear', 'rbf']
            },
            "Naive Bayes": {
                'var_smoothing': [1e-9]
            }
        }
    
    def train_model(self, model_name, use_grid_search=False):
        """Train a specific model with optional hyperparameter tuning"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        param_grid = self.param_grids[model_name]
        
        start_time = time.time()
        
        # For faster training, use the default model parameters without grid search
        # This can be enabled via the use_grid_search parameter if needed
        if use_grid_search and model_name != "Naive Bayes":
            try:
                # Use simplified cross-validation for faster results
                if model_name in ["Random Forest", "SVM"]:
                    search = RandomizedSearchCV(
                        model, param_grid, n_iter=2, cv=2, scoring='accuracy', 
                        random_state=42, n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        model, param_grid, cv=2, scoring='accuracy', n_jobs=-1
                    )
                
                search.fit(self.X_train, self.y_train)
                best_model = search.best_estimator_
                
                # Store best model
                self.trained_models[model_name] = best_model
            except Exception as e:
                # Fallback to simple training if grid search fails
                print(f"Grid search for {model_name} failed: {str(e)}. Using default model.")
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model
        else:
            # Simple training for all models for much faster results
            try:
                # For logistic regression, increase max_iter to avoid convergence warnings
                if model_name == "Logistic Regression":
                    model.set_params(max_iter=2000)
                
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model
            except Exception as e:
                raise ValueError(f"Failed to train {model_name}: {str(e)}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        return self.trained_models[model_name], training_time
    
    def train_all_models(self, use_grid_search=True):
        """Train all models"""
        results = {}
        for model_name in self.models.keys():
            model, training_time = self.train_model(model_name, use_grid_search)
            results[model_name] = {
                "model": model,
                "training_time": training_time
            }
        
        return results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for the specified model"""
        if model_name not in self.trained_models or self.trained_models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        
        if model_name == "Logistic Regression":
            # For logistic regression, use coefficients as importance
            importances = np.abs(model.coef_[0])
            return dict(zip(self.feature_names, importances))
        
        elif model_name in ["Decision Tree", "Random Forest"]:
            # For tree-based models, use feature_importances_
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        
        elif model_name == "SVM" and model.kernel == "linear":
            # For linear SVM, use coefficients
            importances = np.abs(model.coef_[0])
            return dict(zip(self.feature_names, importances))
        
        else:
            raise ValueError(f"Feature importance not available for {model_name}")
