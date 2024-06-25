import numpy as np
import math
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# 評価
from sklearn import metrics
# グリットサーチ
from sklearn.model_selection import GridSearchCV



df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')


df_train = df_train.drop('Ticket', axis=1)


df_train['Fare'] =  [int(i*100) for i in df_train['Fare']]

df_train['Name'] = [re.sub(',.*','',s) for s in df_train['Name'].values]

df_train['Cabin'] = [re.sub('([0-9]) .*','\1',s) if not isinstance(s, float) else s for s in df_train['Cabin'].values]
df_train['Cabin'] = [re.sub('[A-G] ','',s) if not isinstance(s, float) else s for s in df_train['Cabin'].values]

df_train['Deck'] = [re.sub('[^A-G]*','',s) if not isinstance(s, float) else s for s in df_train['Cabin'].values]

df_train['Cabin'] = [re.sub('[A-Z]','',s) if not isinstance(s, float) else s for s in df_train['Cabin'].values]

df_test['Fare'] =  [int(i*100) if not math.isnan(i)  else i for i in df_test['Fare'].values]

df_test['Name'] = [re.sub(',.*','',s) for s in df_test['Name'].values]

df_test['Cabin'] = [re.sub('([0-9]) .*','\1',s) if not isinstance(s, float) else s for s in df_test['Cabin'].values]
df_test['Cabin'] = [re.sub('[A-G] ','',s) if not isinstance(s, float) else s for s in df_test['Cabin'].values]

df_test['Deck'] = [re.sub('[^A-G]*','',s) if not isinstance(s, float) else s for s in df_test['Cabin'].values]

df_test['Cabin'] = [re.sub('[A-Z]','',s) if not isinstance(s, float) else s for s in df_test['Cabin'].values]




le_sex = LabelEncoder()
le_name = LabelEncoder()
le_embarked = LabelEncoder()
le_deck = LabelEncoder()
le_sex.fit(["male","female"])   #(male, female) = (1,0)
le_name.fit(pd.concat([df_train['Name'], df_test['Name']], axis=0).values) 
le_embarked.fit(df_train['Embarked'].values) #(C, Q, S, non) =(0, 1, 2, 3x)
le_deck.fit(df_train['Deck'].values)

embarked_nan = [i for i, x in enumerate(le_embarked.classes_.tolist()) if isinstance(x, float) if(math.isnan(x)) ][0]
deck_nan = [i for i, x in enumerate(le_deck.classes_.tolist()) if isinstance(x, float) if(math.isnan(x)) ][0]


df_train['Sex'] =  le_sex.transform(df_train['Sex'].values)
df_train['Name'] =  le_name.transform(df_train['Name'].values)


df_train['Embarked'] =  [ float('nan') if s == embarked_nan else s for s in le_embarked.transform(df_train['Embarked'].values) ]

df_train['Deck'] = [float('nan') if ss == deck_nan else ss for ss in le_deck.transform(df_train['Deck'].values)]



df_test['Sex'] =  le_sex.transform(df_test['Sex'].values)
df_test['Name'] =  le_name.transform(df_test['Name'].values)


df_test['Embarked'] =  [ float('nan') if s == embarked_nan else s for s in le_embarked.transform(df_test['Embarked'].values) ]

df_test['Deck'] = [float('nan') if ss == deck_nan else ss for ss in le_deck.transform(df_test['Deck'].values)]


df_train.to_csv('titanic/Re_train.csv')
df_test.to_csv('titanic/Re_test.csv')


toDeck = df_train[['Name','Sex','Fare','Survived','SibSp','Parch','Pclass']]
train_set, test_set = train_test_split(toDeck, test_size=0.2, random_state = 0)

print(train_set,test_set)

#訓練データを説明変数データ(X_train)と目的変数データ(y_train)に分割
X_train = train_set.dropna().drop('Survived', axis=1)
y_train = train_set['Survived'].dropna()


#評価データを説明変数データ(X_train)と目的変数データ(y_train)に分割
X_test = test_set.dropna().drop('Survived', axis=1)
y_test = test_set['Survived'].dropna()

model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(df_test[['Name','Sex','Fare','SibSp','Parch','Pclass']])



result = df_test['PassengerId']

result1 = pd.DataFrame( pred ,columns=["Survived"])

result = pd.concat([result, result1["Survived"]], axis=1)

print(pred)
result.to_csv('titanic/result.csv')







