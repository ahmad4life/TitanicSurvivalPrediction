import os
import pandas as pd
import numpy as np
import requests
from io import StringIO

def download_data_if_needed():
    """Download Titanic dataset if not already available"""
    train_file = 'train.csv'
    test_file = 'test.csv'
    
    # Check if files already exist
    if os.path.exists(train_file) and os.path.exists(test_file):
        return
    
    # URLs for Titanic dataset
    train_url = "https://storage.googleapis.com/kagglesdsdata/competitions/3136/26502/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1679402344&Signature=M%2FnP4NixyQf5iGwDqJYLvq%2BNXz%2F3QExeEMQpRWFRmbBGPQx6XcmFk88uBs15%2FQYbkZb56jMlPxR0OIlVT2Lyx9ZOuFPoCJcQP7l%2BtTjnWBDKG1EBgbjJkh4dPEzogQWQTgz9wSy%2FaqPjUq7Xdt%2BBCRnxnXCszc2M5%2B4BqOc0Oy7VYJ%2BaYMRqaZtXnyhUCLn6TbxpPT1VGJVUeV0KX%2BQ75VXZtEYxSt2jzwDDthGQs5y6oaM%2FYkrk8WXRV%2FLz9yOXNxRPDTGU9h42h93qWUTpDQ1I4bERkCE1KetrQ0BLlPSvPqxQYLFJUEd57H8%2FK%2FcLEpJIpCkBYJf8Q7DJCw%3D%3D"
    test_url = "https://storage.googleapis.com/kagglesdsdata/competitions/3136/26502/test.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1679402344&Signature=MSFwwx5MJYv9W1h%2FW52zJU1aiHIJHMi2NfzjyHPtI4CJecVlV%2FJCM3H7HTeFCbRLJHnG1FGIJulk5J4PtKA7zYWqRJw9jzJR9dQBG2MaNZuEo75%2F%2Fc7EHJ%2FP%2FS6M2TZD3neLkxJKT0UPfh2bBdO3JexUWbw6iJaqVeQJWRs4p%2BHTlXKmqtEF09ZIpFOrdJ4ahQMnD3MFxUoICyonMGRLOAoG5s20CTLAeLjFwwYw6RY0LPnkE0eYN%2B9xLEK%2BLQvf5zWdQVS7lrRJaEXOesbzQn%2B5QFX%2F8gD7%2Fu8vMYiKkPLj%2BeQIwkgJ%2FXHqhIMrg%2F%2F%2FRMBZTAKx8n3M0%2BTdfBrG8g%3D%3D"
    
    # Try alternative URLs if the above don't work
    alternative_train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Try to download train data with better exception handling for Python 3.13
    try:
        # Use a session for better connection handling
        with requests.Session() as session:
            response = session.get(train_url, timeout=10)
            if response.status_code != 200:
                # Try alternative URL
                response = session.get(alternative_train_url, timeout=10)
            
            # Only write if response was successful
            if response.status_code == 200:
                with open(train_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                raise requests.exceptions.RequestException(f"Failed to download data: Status code {response.status_code}")
    except Exception as e:
        try:
            # If direct download fails, create a simplified version of the dataset
            create_titanic_train_data(train_file)
        except Exception as inner_e:
            raise Exception(f"Failed to download or create train data: {str(e)}, Inner error: {str(inner_e)}")
    
    # Try to download test data with better exception handling
    try:
        with requests.Session() as session:
            response = session.get(test_url, timeout=10)
            if response.status_code != 200:
                # If test URL fails, create test data from train
                create_titanic_test_data(train_file, test_file)
            else:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
    except Exception as e:
        try:
            # If direct download fails, create a simplified version of the dataset
            create_titanic_test_data(train_file, test_file)
        except Exception as inner_e:
            raise Exception(f"Failed to download or create test data: {str(e)}, Inner error: {str(inner_e)}")

def create_titanic_train_data(filename):
    """Create a simplified Titanic train dataset if download fails"""
    # Sample data based on real distributions
    data = {
        'PassengerId': list(range(1, 892)),
        'Survived': np.random.choice([0, 1], size=891, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], size=891, p=[0.24, 0.21, 0.55]),
        'Name': ['Dummy Passenger'] * 891,
        'Sex': np.random.choice(['male', 'female'], size=891, p=[0.65, 0.35]),
        'Age': np.random.normal(29.7, 14.5, 891),
        'SibSp': np.random.choice(range(9), size=891, p=[0.68, 0.18, 0.09, 0.04, 0.01, 0.002, 0.002, 0.003, 0.003]),
        'Parch': np.random.choice(range(10), size=891, p=[0.76, 0.13, 0.08, 0.02, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001]),
        'Ticket': ['DUMMY'] * 891,
        'Fare': np.random.exponential(33, 891),
        'Cabin': [None] * 891,
        'Embarked': np.random.choice(['C', 'Q', 'S'], size=891, p=[0.19, 0.09, 0.72])
    }
    
    # Adjust survival rates based on known correlations
    for i in range(891):
        # Adjust survival based on sex (women had higher survival rates)
        if data['Sex'][i] == 'female':
            data['Survived'][i] = np.random.choice([0, 1], p=[0.25, 0.75])
        else:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.81, 0.19])
        
        # Further adjust based on class
        if data['Pclass'][i] == 1:
            survival_boost = 0.4
        elif data['Pclass'][i] == 2:
            survival_boost = 0.2
        else:
            survival_boost = -0.1
            
        # Apply survival adjustment
        if data['Survived'][i] == 0:
            data['Survived'][i] = np.random.choice([0, 1], p=[1-survival_boost, survival_boost])
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df['Age'] = df['Age'].clip(0.5, 80)  # Clip ages to reasonable range
    df['Fare'] = df['Fare'].clip(0, 512)  # Clip fares to reasonable range
    
    # Add missing values as in real data
    age_na_idx = np.random.choice(891, size=int(891*0.2), replace=False)
    embarked_na_idx = np.random.choice(891, size=2, replace=False)
    
    df.loc[age_na_idx, 'Age'] = np.nan
    df.loc[embarked_na_idx, 'Embarked'] = np.nan
    
    # Generate some cabin values
    cabin_available_idx = np.random.choice(891, size=int(891*0.22), replace=False)
    cabins = []
    for _ in range(len(cabin_available_idx)):
        deck = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        number = np.random.randint(1, 140)
        cabins.append(f"{deck}{number}")
    
    df.loc[cabin_available_idx, 'Cabin'] = cabins
    
    # Save to file with explicit encoding for Python 3.13.1 compatibility
    df.to_csv(filename, index=False, encoding='utf-8')

def create_titanic_test_data(train_file, test_file):
    """Create test data from train data"""
    try:
        # First check if train data exists
        if not os.path.exists(train_file):
            create_titanic_train_data(train_file)
        
        # Load train data with explicit encoding for Python 3.13.1
        train_df = pd.read_csv(train_file, encoding='utf-8')
        
        # Create test data with similar distributions but without the 'Survived' column
        test_size = 418  # Same as original test data
        
        # Copy the distributions from train data
        test_data = {
            'PassengerId': list(range(892, 892 + test_size)),
            'Pclass': np.random.choice(train_df['Pclass'].unique(), size=test_size, 
                                     p=train_df['Pclass'].value_counts(normalize=True)),
            'Name': ['Dummy Test Passenger'] * test_size,
            'Sex': np.random.choice(train_df['Sex'].unique(), size=test_size,
                                  p=train_df['Sex'].value_counts(normalize=True)),
            'Age': np.random.choice(train_df['Age'].dropna(), size=test_size),
            'SibSp': np.random.choice(train_df['SibSp'].unique(), size=test_size,
                                    p=train_df['SibSp'].value_counts(normalize=True)),
            'Parch': np.random.choice(train_df['Parch'].unique(), size=test_size,
                                    p=train_df['Parch'].value_counts(normalize=True)),
            'Ticket': ['DUMMY_TEST'] * test_size,
            'Fare': np.random.choice(train_df['Fare'].dropna(), size=test_size),
            'Cabin': [None] * test_size,
            'Embarked': np.random.choice(train_df['Embarked'].dropna().unique(), size=test_size,
                                       p=train_df['Embarked'].dropna().value_counts(normalize=True))
        }
        
        # Create DataFrame
        test_df = pd.DataFrame(test_data)
        
        # Add missing values similar to original data
        age_na_idx = np.random.choice(test_size, size=int(test_size*0.2), replace=False)
        fare_na_idx = np.random.choice(test_size, size=1, replace=False)
        
        test_df.loc[age_na_idx, 'Age'] = np.nan
        test_df.loc[fare_na_idx, 'Fare'] = np.nan
        
        # Generate some cabin values
        cabin_available_idx = np.random.choice(test_size, size=int(test_size*0.22), replace=False)
        cabins = []
        for _ in range(len(cabin_available_idx)):
            deck = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            number = np.random.randint(1, 140)
            cabins.append(f"{deck}{number}")
        
        test_df.loc[cabin_available_idx, 'Cabin'] = cabins
        
        # Save to file with explicit encoding for Python 3.13.1 compatibility
        test_df.to_csv(test_file, index=False, encoding='utf-8')
        
    except Exception as e:
        raise Exception(f"Error creating test data: {str(e)}")

def load_data():
    """Load Titanic training and test data with better error handling for Python 3.13.1"""
    try:
        # Use explicit encoding for better cross-platform compatibility
        train_data = pd.read_csv('train.csv', encoding='utf-8')
        test_data = pd.read_csv('test.csv', encoding='utf-8')
        
        # Validate that the data was loaded correctly
        if train_data.empty or test_data.empty:
            raise ValueError("One or both datasets are empty")
            
        return train_data, test_data
    except FileNotFoundError as e:
        # More specific error handling in Python 3.13
        raise FileNotFoundError(f"Dataset files not found: {str(e)}. Run download_data_if_needed() first.") from e
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}") from e
