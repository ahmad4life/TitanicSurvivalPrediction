import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

class ModelEvaluator:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        """Initialize the ModelEvaluator with models and data"""
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model and return performance metrics"""
        if model is None:
            return None
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Calculate ROC curve and AUC
        try:
            y_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        except:
            fpr, tpr = None, None
            roc_auc = None
        
        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        }
    
    def compare_models(self):
        """Compare all trained models and return performance metrics"""
        results = []
        
        for model_name, model in self.models.items():
            result = self.evaluate_model(model_name, model)
            if result:
                results.append(result)
        
        # Convert to DataFrame for easier manipulation
        metrics_df = pd.DataFrame([
            {
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1_score"],
                "ROC AUC": result["roc_auc"] if result["roc_auc"] is not None else np.nan
            } for result in results
        ], index=[result["model_name"] for result in results])
        
        return metrics_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig = go.Figure()
        
        # Add diagonal line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                line=dict(color="navy", width=1, dash="dash"),
                name="Random Classifier",
                showlegend=True
            )
        )
        
        # Add ROC curves for each model
        for model_name, model in self.models.items():
            result = self.evaluate_model(model_name, model)
            if result and result["fpr"] is not None and result["tpr"] is not None:
                fig.add_trace(
                    go.Scatter(
                        x=result["fpr"], y=result["tpr"],
                        name=f"{model_name} (AUC = {result['roc_auc']:.3f})",
                        mode="lines"
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=800,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a specific model"""
        if model_name not in self.models or self.models[model_name] is None:
            return None
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Create labels for plot
        labels = ['Did Not Survive', 'Survived']
        
        # Create annotation text
        z_text = [
            ['TN: ' + str(cm[0, 0]), 'FP: ' + str(cm[0, 1])],
            ['FN: ' + str(cm[1, 0]), 'TP: ' + str(cm[1, 1])]
        ]
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=False
        ))
        
        # Add annotations
        for i in range(2):
            for j in range(2):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=z_text[i][j],
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        # Update layout
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            xaxis=dict(scaleanchor="y"),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def plot_feature_importance(self, model_name, feature_names):
        """Plot feature importance for a specific model"""
        from model_trainer import ModelTrainer
        
        if model_name not in self.models or self.models[model_name] is None:
            return None
        
        # Get model
        model = self.models[model_name]
        
        # Get feature importance
        if model_name == "Logistic Regression":
            # For logistic regression, use coefficients as importance
            importance = np.abs(model.coef_[0])
        elif model_name in ["Decision Tree", "Random Forest"]:
            # For tree-based models, use feature_importances_
            importance = model.feature_importances_
        elif model_name == "SVM" and hasattr(model, 'coef_'):
            # For linear SVM, use coefficients
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        # Create figure
        fig = px.bar(
            x=sorted_importance[:15],  # Top 15 features
            y=sorted_feature_names[:15],
            orientation='h',
            title=f"Feature Importance - {model_name}",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        
        # Update layout
        fig.update_layout(
            yaxis=dict(autorange="reversed")
        )
        
        return fig
