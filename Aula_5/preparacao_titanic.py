import os
import csv
import numpy as np
from scipy.stats import sem
from sklearn import tree, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut

# 1. Localiza o arquivo titanic3.csv automaticamente (na pasta atual ou em ./dat/)
caminho_csv = './dat/titanic3.csv' if os.path.exists('./dat/titanic3.csv') else 'titanic3.csv'

with open(caminho_csv, 'r', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile, delimiter=',', quotechar='"')
    caracteristicas = np.array(next(leitor))
    titanic_X, titanic_y = [], []
    for linha in leitor:
        if len(linha) > 0 and linha[0] != '':
            titanic_X.append(linha)
            titanic_y.append(linha[1])
    titanic_X = np.array(titanic_X)[:, [0, 4, 3]]
    titanic_y = np.array(titanic_y)

# 2. Pré-processamento (Imputação da idade, LabelEncoder e OneHotEncoder)
idades = titanic_X[:, 1]
idade_med = np.mean(titanic_X[idades != '', 1].astype(float))
titanic_X[titanic_X[:, 1] == '', 1] = idade_med

titanic_X[:, 2] = LabelEncoder().fit_transform(titanic_X[:, 2])

int_classes = LabelEncoder().fit_transform(titanic_X[:, 0]).reshape(-1, 1)
novas_colunas = OneHotEncoder().fit_transform(int_classes).toarray()
titanic_X = np.delete(np.concatenate([titanic_X, novas_colunas], axis=1), [0], 1).astype(float)
titanic_y = titanic_y.astype(float)
caracteristicas = ['age', 'sex', 'first_class', 'second_class', 'third_class']

# 3. Divisão Treino (75%) e Teste (25%)
X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.25, random_state=33)

# 4. Funções auxiliares da aula anterior (loo_cv e measure_performance)
def loo_cv(X_train, y_train, clf):
    loo = LeaveOneOut()
    scores = np.zeros(X_train.shape[0])
    for train_index, test_index in loo.split(X_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        clf_fitted = clf.fit(X_train_cv, y_train_cv)
        y_pred = clf_fitted.predict(X_test_cv)
        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))
    print(f"Acurácia Média (LOO-CV): {np.mean(scores):.3f} (+/-{sem(scores):.3f})")

def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_r2_score=False):
    y_pred = clf.predict(X)
    if show_accuracy:
        print(f"Acurácia: {metrics.accuracy_score(y, y_pred):.3f}\n")
    if show_classification_report:
        print("Relatório de Classificação:")
        print(metrics.classification_report(y, y_pred), "\n")
    if show_confusion_matrix:
        print("Matriz de Confusão:")
        print(metrics.confusion_matrix(y, y_pred), "\n")
    if show_r2_score:
        print(f"Coeficiente de determinação (R²): {metrics.r2_score(y, y_pred):.3f}\n")

print("✅ Dados do Titanic (X_train, X_test, y_train, y_test) e funções carregados com sucesso!")
