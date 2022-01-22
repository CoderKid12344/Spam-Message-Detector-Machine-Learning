import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

X_test = str(input("Enter The Message You want to detect: "))
Train = pd.read_csv('Train.csv')

X_train = list(Train['msg'].values)
Y_train = Train["value"].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform([X_test])

Model = LogisticRegression(max_iter=50000000000000000000)
Model.fit(X_train, Y_train)
Y_Pred = Model.predict(X_test)

if Y_Pred[0] == 'spam':
    print("This message is a spam")
elif Y_Pred[0] == 'ham':
    print("This message is not a spam")
else:
    print("No Data Found!")
