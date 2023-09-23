import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('seattle-weather.csv')

df.info()

df.drop(['date'], axis=1, inplace=True)

print(df['weather'].value_counts())

def set_weather(weather):
    if weather == 'sun':
        return 0

    if weather == 'fog':
        return 1

    if weather == 'drizzle':
        return 2

    if weather == 'rain':
        return 3

    if weather == 'snow':
        return 4

df['weather'] = df['weather'].apply(set_weather)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop("weather", axis=1)
Y = df["weather"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifer = KNeighborsClassifier()
classifer.fit(X_train, Y_train)

pred = classifer.predict(X_test)

print(accuracy_score(Y_test, pred))
print(confusion_matrix(Y_test, pred))