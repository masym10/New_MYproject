import pandas as pd
import matplotlib.pyplot as plt

def counts(name_colums):
    print(df[name_colums].value_counts())


df = pd.read_csv('train.csv')

df['CryoSleep'].fillna(False, inplace=True)
df['HomePlanet'].fillna('Earth', inplace=True)
df['VIP'].fillna(False, inplace=True)
df['Destination'].fillna('TRAPPIST-1e', inplace=True)
df['Age'].fillna(28, inplace=True)
df.fillna(0, inplace=True)

df[list(pd.get_dummies(df['HomePlanet']).columns)] = pd.get_dummies(df['HomePlanet'])
df[list(pd.get_dummies(df['Destination']).columns)] = pd.get_dummies(df['Destination'])

df.drop([ 'HomePlanet', 'Name', 'Cabin', 'Destination', 'PassengerId'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop("Transported", axis=1)
Y = df["Transported"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.50)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifer = KNeighborsClassifier()
classifer.fit(X_train, Y_train)

pred = classifer.predict(X_test)

print(accuracy_score(Y_test, pred))
print(confusion_matrix(Y_test, pred))
