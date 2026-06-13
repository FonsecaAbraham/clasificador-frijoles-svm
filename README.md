# Proyecto de Clasificación de Variedades de Frijoles Secos

### **Descripción del Proyecto**

Este proyecto implementa algoritmos de Machine Learning para clasificar diferentes variedades de frijoles secos utilizando sus características morfométricas. El objetivo es desarrollar un modelo capaz de distinguir con precisión entre las distintas variedades, lo cual tiene aplicaciones importantes en agricultura y control de calidad.

### **Objetivos**

#### **Exploración de Datos:** Entender la distribución y relaciones de las características morfométricas de los frijoles.

#### **Preprocesamiento de Datos:** Preparar el dataset mediante normalización para optimizar el rendimiento del modelo.

#### **Modelado de Clasificación:** Desarrollar y entrenar un modelo de Support Vector Machine (SVM) para la clasificación.

#### **Evaluación del Modelo:** Analizar el rendimiento del modelo utilizando métricas clave y visualizaciones como curvas de aprendizaje y matrices de confusión.

### **Metodología**

#### **1. Carga y Exploración de Datos**
El dataset utilizado proviene del repositorio UCI Machine Learning y contiene 13,611 instancias con 16 características morfométricas (como Área, Perímetro, Longitud del Eje Mayor, etc.) y una variable objetivo que representa 7 clases diferentes de frijoles secos (SEKER, BARBUNYA, BOMBAY, CALI, SIRA, DERMASON, HOROZ). Se confirmó la ausencia de valores nulos.

#### **2. Preprocesamiento de Datos**
Debido a las diferentes magnitudes de las variables de entrada, se aplicó un escalado estándar (StandardScaler) a las características para asegurar que ninguna característica dominara el proceso de entrenamiento del modelo. Los datos se dividieron en conjuntos de entrenamiento y prueba (80% y 20% respectivamente) utilizando train_test_split con random_state=22 para reproducibilidad.

#### **3. Modelado (SVM)**
Se seleccionó un modelo de Support Vector Machine (SVM) debido a su eficacia en problemas de clasificación no lineal y su capacidad para manejar datos morfométricos. Para optimizar el rendimiento, se realizó una búsqueda de hiperparámetros utilizando GridSearchCV con validación cruzada estratificada (StratifiedKFold, n_splits=10).

Los hiperparámetros explorados fueron:

C: [1, 5, 10, 15, 20, 30]

gamma: [0.02, 0.05, 0.08, 0.1, 0.15]

kernel: ['rbf']

#### **4. Evaluación del Modelo**
El modelo óptimo encontrado por GridSearchCV se evaluó utilizando varias métricas y visualizaciones:

Mejor Exactitud (en entrenamiento con validación cruzada): 92.9738%

Mejores Parámetros: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}

Exactitud en el Conjunto de Prueba: 93.6100%

Curva de Aprendizaje

La curva de aprendizaje mostró una convergencia adecuada, indicando un buen equilibrio entre sesgo y varianza, y que el modelo no sufre de sobreajuste significativo ni de subajuste.

Matriz de Confusión Híbrida
La matriz de confusión proporcionó una vista detallada del rendimiento del modelo para cada clase, mostrando las predicciones correctas e incorrectas, tanto en valores absolutos como en proporciones. Se observó una alta precisión y recall en la mayoría de las clases.

### **Reporte de Clasificación**
El reporte de clasificación detalló la precisión, recall y f1-score para cada una de las 7 clases, así como el soporte. Las métricas fueron consistentemente altas, superando el 90% para la mayoría de las clases, lo que valida la robustez del modelo.

### **Conclusión**
El modelo SVM optimizado demostró ser altamente efectivo en la clasificación de variedades de frijoles secos, alcanzando una exactitud superior al 93% en el conjunto de prueba. Los resultados de la evaluación, incluyendo la curva de aprendizaje, la matriz de confusión y el reporte de clasificación, confirman la solidez y fiabilidad del modelo.

Este proyecto establece una base sólida para futuras investigaciones, que podrían incluir la exploración de otros algoritmos, la ingeniería de características avanzadas o la implementación de técnicas de aprendizaje profundo para comparar y mejorar el rendimiento.
