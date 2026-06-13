from src.data_processing import load_dry_bean_data, preprocess_data
from src.model_training import train_svm_gridsearch
from src.evaluation import (
    print_basic_metrics, 
    generate_classification_report,
    plot_learning_curve, 
    plot_hybrid_confusion_matrix
)

def main():
    print("1. Cargando datos...")
    df = load_dry_bean_data()
    
    print("2. Preprocesando y escalando características...")
    X_train_norm, X_test_norm, y_train, y_test, classes = preprocess_data(df)
    
    print("3. Entrenando modelo SVM con GridSearchCV...")
    grid_model = train_svm_gridsearch(X_train_norm, y_train)
    
    print("4. Evaluando resultados...")
    y_pred = print_basic_metrics(grid_model, X_test_norm, y_test)
    generate_classification_report(y_test, y_pred, grid_model.classes_)
    
    print("5. Generando visualizaciones...")
    plot_learning_curve(grid_model.best_estimator_, X_train_norm, y_train)
    plot_hybrid_confusion_matrix(y_test, y_pred, grid_model.classes_)

if __name__ == "__main__":
    main()
