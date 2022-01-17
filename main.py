import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, hamming_loss, confusion_matrix, \
    average_precision_score
from sklearn.model_selection import train_test_split


def get_dataset_from_label(df, label):
    return_df = pd.DataFrame(columns=["text", "toxic"])
    return_df["text"] = df["text"]
    return_df["toxic"] = df.apply(lambda x: 1 if x[label] == 1 else 0, axis=1)

    return return_df


multilabel_data = pd.read_csv('corpus/multilabel_grouped.csv')
multilabel_data = multilabel_data[["text", "homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]]

embeddings = ['elmo']
avg_precision = []
hammingloss_scores = []

for e in embeddings:
    for label in multilabel_data.columns[1:]:

        df = get_dataset_from_label(multilabel_data, label)
        _, test_df = train_test_split(df, train_size=0.9, random_state=2021)

        train = pd.read_csv(os.path.join(e, label+'_train.csv'))
        test = pd.read_csv(os.path.join(e, label + '_test.csv'))

        clf = GradientBoostingClassifier(n_estimators=500, max_depth=5, random_state=2021)
        clf.fit(train[['toxic', 'nontoxic']], np.ravel(train[['label']]))
        y_pred = clf.predict(test[['toxic', 'nontoxic']])

        print(classification_report(test_df['toxic'], y_pred))
        tn, fp, fn, tp = confusion_matrix(test_df['toxic'], y_pred).ravel()
        print(f'FN: {fn}')
        print(f'FP: {fp}')
        print(f'TP: {tp}')
        print(f'TN: {tn}')
        avg_precision.append(average_precision_score(test_df["toxic"], y_pred))
        hammingloss_scores.append(hamming_loss(test_df["toxic"], y_pred))
    print(f"Avg Precision: {np.mean(avg_precision)}")
    print(f"Avg Macro Hamming Loss: {np.mean(hammingloss_scores)}")
    avg_precision, hammingloss_scores = [], []
