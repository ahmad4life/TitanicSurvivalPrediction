import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, train_data, test_data):
        """Initialize DataProcessor with training and test data"""
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.preprocessing_pipeline = None
        
        # Default options
        self.options = {
            'age_strategy': 'median',
            'embarked_strategy': 'most_frequent',
            'create_family_size': True,
            'create_title_feature': True,
            'create_age_groups': True,
        }
    
    def set_options(self, **kwargs):
        """Set preprocessing options"""
        for key, value in kwargs.items():
            if key in self.options:
                self.options[key] = value
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Handle missing Age values
        age_imputer = SimpleImputer(strategy=self.options['age_strategy'])
        self.train_data['Age'] = age_imputer.fit_transform(self.train_data[['Age']])
        self.test_data['Age'] = age_imputer.transform(self.test_data[['Age']])
        
        # Handle missing Embarked values
        if self.options['embarked_strategy'] == 'most_frequent':
            most_common_embarked = self.train_data['Embarked'].mode()[0]
            self.train_data['Embarked'] = self.train_data['Embarked'].fillna(most_common_embarked)
            self.test_data['Embarked'] = self.test_data['Embarked'].fillna(most_common_embarked)
        else:  # 'drop'
            self.train_data = self.train_data.dropna(subset=['Embarked'])
        
        # Handle missing Fare values
        fare_imputer = SimpleImputer(strategy='median')
        self.train_data['Fare'] = fare_imputer.fit_transform(self.train_data[['Fare']])
        self.test_data['Fare'] = fare_imputer.transform(self.test_data[['Fare']])
        
        # Handle missing Cabin values - Create a new feature HasCabin
        self.train_data['HasCabin'] = self.train_data['Cabin'].notna().astype(int)
        self.test_data['HasCabin'] = self.test_data['Cabin'].notna().astype(int)
    
    def engineer_features(self):
        """Create new features from existing data"""
        # Process both train and test data
        for data in [self.train_data, self.test_data]:
            # Create Family Size feature
            if self.options['create_family_size']:
                data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
                data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
            
            # Extract title from name
            if self.options['create_title_feature']:
                data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
                # Group rare titles
                rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
                data['Title'] = data['Title'].replace(rare_titles, 'Rare')
                data['Title'] = data['Title'].replace('Mlle', 'Miss')
                data['Title'] = data['Title'].replace('Ms', 'Miss')
                data['Title'] = data['Title'].replace('Mme', 'Mrs')
            
            # Create age groups
            if self.options['create_age_groups']:
                data['AgeGroup'] = pd.cut(
                    data['Age'], 
                    bins=[0, 5, 12, 18, 35, 60, 100],
                    labels=['Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
                )
            
            # Create Fare categories
            data['FareCategory'] = pd.qcut(
                data['Fare'], 
                q=4, 
                labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
            
            # Extract deck from cabin
            data['Deck'] = data['Cabin'].astype(str).str[0]
            data['Deck'] = data['Deck'].replace(['n', 'N'], 'U')  # Replace NaN deck with 'U'
            data['Deck'] = data['Deck'].replace(['T'], 'U')  # Replace rare deck 'T' with 'U'
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        # Define categorical and numerical features
        categorical_features = ['Sex', 'Embarked', 'Deck']
        
        if self.options['create_title_feature']:
            categorical_features.append('Title')
        
        if self.options['create_age_groups']:
            categorical_features.append('AgeGroup')
            
        categorical_features.append('FareCategory')
        
        numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'HasCabin']
        
        if self.options['create_family_size']:
            numerical_features.extend(['FamilySize', 'IsAlone'])
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Store feature names for later use
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
    
    def split_data(self):
        """Split the data into training and test sets"""
        X = self.train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        y = self.train_data['Survived']
        
        # Get feature names before splitting
        self.feature_names = self._get_feature_names(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        # Fit to X_train and transform both X_train and X_test
        self.X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)
        self.X_test_processed = self.preprocessing_pipeline.transform(X_test)
        
        # Update X_train and X_test to processed versions
        self.X_train = self.X_train_processed
        self.X_test = self.X_test_processed
        
        # Update feature names
        self.feature_names = self._get_feature_names_after_preprocessing()
        
        return self.X_train, self.X_test
    
    def _get_feature_names(self, X):
        """Get feature names before preprocessing"""
        return list(X.columns)
    
    def _get_feature_names_after_preprocessing(self):
        """Get feature names after preprocessing with OneHotEncoder"""
        feature_names = []
        
        # Get numerical feature names
        feature_names.extend(self.numerical_features)
        
        # Get categorical feature names with one-hot encoding
        one_hot_encoder = self.preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = one_hot_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(categorical_feature_names)
        
        return feature_names
    
    def preprocess_new_passenger(self, passenger_data):
        """Preprocess new passenger data for prediction"""
        # Check if preprocessing pipeline exists
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not initialized. Run encode_categorical_features and scale_features first.")
            
        # Convert passenger data to DataFrame
        passenger_df = pd.DataFrame([passenger_data])
        
        # Apply feature engineering
        # Handle HasCabin
        passenger_df['HasCabin'] = passenger_df['Cabin'].notna().astype(int)
        
        # Create FamilySize and IsAlone if enabled
        if self.options['create_family_size']:
            passenger_df['FamilySize'] = passenger_df['SibSp'] + passenger_df['Parch'] + 1
            passenger_df['IsAlone'] = (passenger_df['FamilySize'] == 1).astype(int)
        
        # Extract title from name if enabled
        if self.options['create_title_feature']:
            passenger_df['Title'] = passenger_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            # Handle missing titles or unusual formats
            if passenger_df['Title'].isna().any():
                passenger_df['Title'] = passenger_df['Title'].fillna('Mr')
                
            # Map titles to the same categories used in training
            rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
            passenger_df['Title'] = passenger_df['Title'].replace(rare_titles, 'Rare')
            passenger_df['Title'] = passenger_df['Title'].replace('Mlle', 'Miss')
            passenger_df['Title'] = passenger_df['Title'].replace('Ms', 'Miss')
            passenger_df['Title'] = passenger_df['Title'].replace('Mme', 'Mrs')
        
        # Create age groups if enabled
        if self.options['create_age_groups']:
            passenger_df['AgeGroup'] = pd.cut(
                passenger_df['Age'], 
                bins=[0, 5, 12, 18, 35, 60, 100],
                labels=['Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
            )
        
        # Create Fare categories
        passenger_df['FareCategory'] = pd.cut(
            passenger_df['Fare'], 
            bins=[-1, 7.91, 14.454, 31, 512.329], 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # Extract deck from cabin
        passenger_df['Deck'] = passenger_df['Cabin'].astype(str).str[0]
        passenger_df['Deck'] = passenger_df['Deck'].replace(['n', 'N'], 'U')
        passenger_df['Deck'] = passenger_df['Deck'].replace(['T'], 'U')
        
        # Drop columns not used in the model
        passenger_df = passenger_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
        
        try:
            # Transform using the preprocessing pipeline
            passenger_processed = self.preprocessing_pipeline.transform(passenger_df)
            return passenger_processed
        except Exception as e:
            # Provide more helpful error message
            raise ValueError(f"Error preprocessing passenger data: {str(e)}. Make sure preprocessing has been completed first.") from e
