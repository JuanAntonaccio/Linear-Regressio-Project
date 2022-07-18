# Trabajamos con el dataset de costos de seguros
# Los datos estan limpios no hay nulos
# Este proceso lo hacemos luego de trabajar en el notebok



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')

#Label encode categorical features

from sklearn.preprocessing import LabelEncoder
# Vemos para la variable sexo que en este caso es sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# Vemos para la variable fumador que es fumador o no
# en este caso se llama smoker 
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
# Vemos para la variable region
# en este caso se llama region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)

# Vamos a ver que tanta correlacion tienen las variables o features con la variable objetivo charges

df.corr()['charges'].sort_values()

f, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Vamos a predecir el seguro, con la regresion lineal

x = df.drop(['charges'], axis = 1)
y = df.charges

X_train,X_test,y_train,y_test = train_test_split(x,y, random_state = 121)
lr = LinearRegression().fit(X_train,y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(lr.score(X_test,y_test))

# Ahora borramos las regiones y aplicamos el polinomial

x = df.drop(['charges','region'], axis = 1)
y = df.charges

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(x)

X_train,X_test,y_train,y_test = train_test_split(x_quad,y, random_state = 121)

plr = LinearRegression().fit(X_train,y_train)

y_train_pred = plr.predict(X_train)
y_test_pred = plr.predict(X_test)

print(plr.score(X_test,y_test))

import joblib

#save your model or results
joblib.dump(plr, '../models/modelo_entrenado_seguros.pkl')


