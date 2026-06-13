import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_dry_bean_data() -> pd.DataFrame:
    """Ingesta del dataset Dry Bean desde UCIML."""
    dry_bean = fetch_ucirepo(id=602)
    df = dry_bean.data.features.join(dry_bean.data.targets)
    return df

def preprocess_data(
    df: pd.DataFrame, 
    target_col: str = "Class", 
    test_size: float = 0.2, 
    random_state: int = 22
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, np.ndarray]:
    """Separa variables dependientes/independientes, particiona y escala los datos."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Se retorna un array de clases únicas para uso en métricas posteriores
    classes = y.unique()
    
    return X_train_norm, X_test_norm, y_train, y_test, classes
