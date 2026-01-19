import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data frame
df = sns.load_dataset('iris')

print(df.head())
print(df.shape)

features = df.drop('species', axis=1).columns
X = df[features]
y = df['species']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_val.shape)
print(X_train.shape)


model = RandomForestClassifier()
model.fit(X_train, y_train, )

