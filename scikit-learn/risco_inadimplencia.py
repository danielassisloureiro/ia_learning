import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Simulando dados
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'idade': np.random.randint(18, 65, n),
    'salario_mensal': np.random.normal(5000, 2000, n).clip(1000),
    'tempo_empresa': np.random.randint(1, 240, n),
    'qtd_dividas_anteriores': np.random.poisson(1.5, n),
    'score_credito': np.random.randint(300, 1000, n),
})

# Regra oculta para gerar a label
df['inadimplente'] = (
    (df['score_credito'] < 500) |
    (df['qtd_dividas_anteriores'] > 2)
).astype(int)

# Separando features e label
X = df.drop('inadimplente', axis=1)
y = df['inadimplente']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliando
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importância das features
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nImportância das features:\n", importances.sort_values(ascending=False))