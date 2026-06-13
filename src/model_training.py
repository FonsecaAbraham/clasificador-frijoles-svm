import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

def train_svm_gridsearch(
    X_train: np.ndarray, 
    y_train: pd.Series, 
    random_state: int = 22
) -> GridSearchCV:
    """Configura y ejecuta la búsqueda en cuadrícula para el modelo SVM."""
    cv_random = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    param_grid = {
        'C': [1, 5, 10, 15, 20, 30],
        'gamma': [0.02, 0.05, 0.08, 0.1, 0.15],
        'kernel': ['rbf']
    }
    
    grid = GridSearchCV(
        SVC(),
        param_grid,
        refit=True,
        verbose=1,
        cv=cv_random,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    return grid
