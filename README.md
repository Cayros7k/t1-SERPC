# README - Processo de Normalização e Otimização de Hiperparâmetros para KNN

Este README descreve o processo de normalização de dados e otimização de hiperparâmetros para um modelo K-Nearest Neighbors (KNN) utilizando a biblioteca scikit-learn em Python.

## NORMALIZAÇÃO DOS DADOS

A normalização é importante no pré-processamento de dados, especialmente para o KNN. A seguir, o processo de normalização utilizado:

``` 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
``` 

- Importei a classe `StandardScaler` de `sklearn.preprocessing`.
- Iniciei um objeto `scaler` de `StandardScaler`.
- Utilizei o método `fit_transform()` para ajustar o scaler aos dados (`fit`) e normalizá-los (`transform`).

## OTIMIZAÇÃO DE HIPERPARÂMETROS

A otimização de hiperparâmetros é realizada para encontrar melhores parâmetros para o modelo, a fim de maximizar seu desempenho. A seguir, o processo de otimização de hiperparâmetros utilizado:

```
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled, y)

best_params = grid_search.best_params_
```

- Importei a classe `GridSearchCV` de `sklearn.model_selection`.
- Defini um grid de hiperparâmetros para pesquisa.
- Utilizei o `GridSearchCV` para ajustar o modelo KNN com diferentes combinações de hiperparâmetros, usando validação cruzada.
- Obtive os melhores hiperparâmetros com o atributo `best_params_` do objeto `grid_search`.

## TREINAMENTO E AVALIAÇÃO DO MODELO

Por fim, treinei o modelo final com os melhores hiperparâmetros e avaliei sua performance. A seguir, o processo:

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
best_knn.fit(X_scaled, y)

y_preds = best_knn.predict(X_scaled)

accuracy_score(y, y_preds)
```

- Importei a classe `KneighborsClassifier` de `sklearn.neighbors` e a função `accuracy_score` de `sklearn.metris`.
- Criei o modelo KNN final com os melhores hiperparâmetros encontrados.
- Fiz previsões com o modelo final.
- Calculei a precisão das previsões comparando os rótulos reais com os previstos.

> Henrique de Oliveira Silveira | Sistemas de Informação | 7° Período | Noturno


