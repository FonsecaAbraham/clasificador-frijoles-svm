import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold

def print_basic_metrics(grid_model, X_test: np.ndarray, y_test: pd.Series) -> np.ndarray:
    """Imprime la exactitud básica y retorna las predicciones."""
    print(f"Mejor Exactitud (CV): {grid_model.best_score_:.4%}")
    print(f"Mejores Parámetros: {grid_model.best_params_}")
    
    y_pred = grid_model.predict(X_test)
    print(f"Exactitud en prueba: {accuracy_score(y_test, y_pred):.4%}\n")
    return y_pred

def generate_classification_report(y_test: pd.Series, y_pred: np.ndarray, classes: np.ndarray) -> None:
    """Imprime el reporte detallado multiclase."""
    reporte = classification_report(y_test, y_pred, target_names=classes)
    print("Reporte de Rendimiento por Clase:\n")
    print(reporte)

def plot_learning_curve(model, X_train: np.ndarray, y_train: pd.Series, random_state: int = 22) -> None:
    """Genera y muestra la curva de aprendizaje del modelo."""
    cv_random = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv_random, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy",
        shuffle=True, random_state=random_state
    )

    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Entrenamiento")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validación Cruzada")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title("Curva de Aprendizaje: SVM Optimizado")
    plt.xlabel("Tamaño del Set de Entrenamiento")
    plt.ylabel("Exactitud (Accuracy)")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_hybrid_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, classes: np.ndarray) -> None:
    """Genera y muestra la matriz de confusión híbrida."""
    cm_abs = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')

    etiquetas_texto = np.asarray([
        f"{val_norm:.2f}\n(n={val_abs})" if val_abs > 0 else ""
        for val_norm, val_abs in zip(cm_norm.flatten(), cm_abs.flatten())
    ]).reshape(cm_abs.shape)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=etiquetas_texto, fmt='', cmap='Blues',
        xticklabels=classes, yticklabels=classes, vmin=0.0, vmax=1.0,
        cbar_kws={'label': 'Proporción de Acierto'}
    )

    plt.title("Matriz de Confusión Híbrida: Evaluación SVM", fontsize=16, pad=20)
    plt.xlabel("Predicción de clasificación", fontsize=14, labelpad=10)
    plt.ylabel("Clasificación Real", fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
