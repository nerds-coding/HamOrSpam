import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize, sent_tokenize
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('wordnet')


class HamSpam:

    def __init__(self):
        self.data = pd.read_csv('SpamOrHam.csv', encoding='ISO-8859-1')
        self.data = self.data[['v1', 'v2']]
        self.data.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
        self.data['Label'] = self.data['Label'].map({'ham': 0, 'spam': 1})

    def get_wordnet_pos(self, pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def CleaningData(self, val):
        rex = re.sub(r'[^a-zA-Z0-9]+', ' ', val)

        pos = pos_tag(word_tokenize(rex))

        filter = [WordNetLemmatizer().lemmatize(x[0], HamSpam.get_wordnet_pos(self, x[1]))
                  for x in pos if x[0] not in stopwords.words('english')]

        filter = ' '.join(filter)

        return filter

    def FilterData(self):
        self.data['ppData'] = 0
        for x in range(len(self.data)):
            self.data.iloc[x, 2] = HamSpam.CleaningData(self, self.data.iloc[x, 1])

        tf = TfidfVectorizer(max_features=500, ngram_range=(1, 4), lowercase=True)
        vals = tf.fit_transform(self.data['ppData'])

        xTrain, xTest, yTrain, yTest = train_test_split(
            vals.toarray(), self.data['Label'], test_size=0.2, random_state=1)
        xTrain, yTrain = SMOTE(k_neighbors=4).fit_resample(xTrain, yTrain)

        models = {

            'RF': {
                "Model": RandomForestClassifier(),
                "Params": {
                    'n_estimators': np.linspace(50, 500, 50, dtype='int64'),
                    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
                }},

            'svc': {
                "Model": SVC(),
                "Params": {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                           'gamma': np.arange(1e-4, 1e-2, 0.0001),
                           'C': np.linspace(1, 100, 10),
                           'degree': np.linspace(1, 10, 10)
                           }},



            'log': {
                'Model': LogisticRegression(),
                'Params': {
                    'C': np.arange(0.1, 1.0, 0.1),
                }
            },



            'DT': {
                'Model': DecisionTreeClassifier(),
                'Params': {
                    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
                }
            }
        }

        res = {}
        for key, values in models.items():
            grid = RandomizedSearchCV(
                estimator=values['Model'], param_distributions=values['Params'], cv=5, return_train_score=False)
            grid = grid.fit(xTrain, yTrain)
            res[key] = {'Best Params': grid.best_params_, 'Best Score': grid.best_score_,
                        'Mean Score': grid.cv_results_["mean_test_score"].mean()}

        with open('BestParams.txt', 'w')as file:
            for key, values in res.items():
                file.write('%s:%s\n' % (key, values))

    def AccuracyTest(self, predData, yTestData):
        print("Accuracy Test ", accuracy_score(yTestData, predData))
        print()
        print("Precision Test ", precision_score(yTestData, predData))
        print()
        print("F1 Score ", f1_score(yTestData, predData))
        print()
        print("Confusion Score ")
        print(confusion_matrix(yTestData, predData))
        print()
        print("Classification Report Score ")
        print(classification_report(yTestData, predData))

    def FinalPrediction(self):
        FinalModel = {'dt': DecisionTreeClassifier(criterion='entropy',
                                                   max_depth=7,
                                                   min_samples_leaf=3,
                                                   min_samples_split=3),

                      'svm':  SVC(C=45.0,
                                  degree=3.0,
                                  gamma=0.007300000000000001,
                                  kernel='rbf'),

                      'RF': RandomForestClassifier(criterion='entropy',
                                                   max_depth=9,
                                                   min_samples_leaf=9,
                                                   min_samples_split=3,
                                                   n_estimators=435),

                      'log': LogisticRegression(C=0.9, penalty='l2'),

                      'NB': MultinomialNB()

                      }
        for key, values in FinalModel.items():
            ada = AdaBoostClassifier(base_estimator=values, n_estimators=100, algorithm='SAMME')
            vals = ada.fit(xTrain, yTrain)

            print()
            print(key)

            filename = 'Adaboost'+key+'.sav'
            pickle.dump(vals, open(filename, 'wb'))
            loaded = pickle.load(open(filename, 'rb'))
            HamSpam.AccuracyTest(self, loaded.predict(xTest), yTest)

            print('---'*30)


ham = HamSpam()
#ham.FilterData()
#ham.FinalPrediction()
