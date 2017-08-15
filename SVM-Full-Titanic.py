import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

#import
train = pd.read_csv("//tech1/vol1/Interns/Will/Titanic Dataset/train.csv")
test = pd.read_csv("//tech1/vol1/Interns/Will/Titanic Dataset/test.csv")

full = [train, test]
full = pd.concat(full)

full['Age'] = full["Age"].fillna(full["Age"].median())
full['Fare'] = full["Fare"].fillna(full["Fare"].median())

lb = preprocessing.LabelBinarizer()


full['Sex'] = lb.fit_transform(full['Sex'])
test['Sex'] = lb.fit_transform(test['Sex'])

full["ParentsAndChildren"] = full["Parch"]
full["SiblingsAndSpouses"] = full["SibSp"]



from sklearn import svm

test['Age'] = test["Age"].fillna(test["Age"].median())
test['Fare'] = test["Fare"].fillna(test["Fare"].median())

X = np.array(full[['Age', 'Fare', 'ParentsAndChildren', 'SiblingsAndSpouses', 'Pclass', 'Sex']])
y = np.array(full['Survived'])

clf = svm.SVC()

clf.fit(X,y)

test2 = test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Sex']]
testSurvived = test[['Survived']]

result = clf.predict(test2)
result_array = np.array(result)

score = clf.score(test2, testSurvived)
print(score)



submission = pd.DataFrame(result, test['PassengerId'])


submission.to_csv("C:/Users/whrice.LCCA/Desktop/results.csv")

