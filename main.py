import json
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()


def read_file(path, encoding_scheme):
    with open(path, 'r', encoding = encoding_scheme) as f:
      data = json.load(f)
    data_df = pd.DataFrame(data)
    return data_df

def get_data(data_df, column_name):
    df = data_df[['id','confirmation_bias', column_name]].copy()
    df['text_id'] = data_df['id'].copy()
    df = df.explode(column_name)
    df['span'] = df[column_name].apply(lambda x: x['span'])
    df[column_name] =  df[column_name].apply(lambda x: x['text'])
    return df

def get_data_para(data_df):
    para_df = data_df[['id', 'confirmation_bias','paragraphs']].copy()
    para_df = para_df.explode('paragraphs')
    para_df['sufficient'] =  para_df['paragraphs'].apply(lambda x: x['sufficient'])
    para_df['paragraphs'] =  para_df['paragraphs'].apply(lambda x: x['text'])
    return para_df

def assign_scores(x):
    res = analyzer.polarity_scores(x)
    return list(res.values())

def sentiment_analysis(df, column_name):
    df['scores'] = df[column_name].apply(assign_scores)
    df[['neg', 'neu', 'pos', 'comp']] = pd.DataFrame(df.scores.tolist(), index = df.index)
    return df
    
def train_test_split(path):
    return train, test

def export_prediction(y_test, y_pred):
    df1 = pd.DataFrame(y_test)
    df1['new_index'] = range(0, len(df1))
    df2 = pd.DataFrame(data=y_pred, columns=['confirmation_bias_pred'])
    output = pd.merge(df1, df2, right_index=True, left_on='new_index')
    output = output.drop(['confirmation_bias', 'new_index'], axis=1)

    output.reset_index(inplace=True)
    output = output.rename(columns = {'index':'id', 'confirmation_bias_pred':'confirmation_bias' })
    output['id'] = output['id'].astype('string')

    output.to_json('predictions.json', orient='records', indent=3)
    return

def cross_validation(df, clf):
    x = df[['neg','pos','neu','comp']]
    y = df['confirmation_bias']
    scores = cross_val_score(clf, x, y,scoring="f1",cv=10)
    return scores

def main():
    path = "data/essay-corpus.json"
    data_df = read_file(path, 'latin-1')

    claims_df = get_data(data_df, 'claims')
    majclaims_df = get_data(data_df, 'major_claim')
    premises_df = get_data(data_df, 'premises')
    para_df = get_data_para(data_df)

    claims_df = sentiment_analysis(claims_df, 'claims')
    majclaims_df = sentiment_analysis(majclaims_df, 'major_claim')
    premises_df = sentiment_analysis(premises_df, 'premises')

    claims = claims_df[['text_id', 'neg', 'neu', 'pos', 'comp']].groupby(['text_id']).sum()
    maj_claims = majclaims_df[['text_id', 'neg', 'neu', 'pos', 'comp']].groupby(['text_id']).sum() 
    premises = premises_df[['text_id', 'neg', 'neu', 'pos', 'comp']].groupby(['text_id']).sum()

    df1 = pd.merge(claims, maj_claims, on='text_id')
    df2 = pd.merge(df1, premises, on='text_id')
    final_df = pd.merge(df2, data_df[['id', 'confirmation_bias']], left_on='text_id', right_on='id')

    final_df['neg'] = (final_df['neg'] + final_df['neg_x'] + final_df['neg_y']) / 3
    final_df['pos'] = (final_df['pos'] + final_df['pos_x'] + final_df['pos_y']) / 3
    final_df['neu'] = (final_df['neu'] + final_df['neu_x'] + final_df['neu_y']) / 3
    final_df['comp'] = (final_df['comp'] + final_df['comp_x'] + final_df['comp_y']) / 3

    test_file = pd.read_csv('data/train-test-split.csv', sep=";")
    test_file['id'] = test_file.index + 1
    final_df = pd.merge(final_df, test_file, on='id')
    train = final_df[final_df['SET']=='TRAIN'][['neg','pos','neu','comp', 'confirmation_bias']]
    test = final_df[final_df['SET']=='TEST'][['neg','pos','neu','comp', 'confirmation_bias']]

    X_train = train[['neg','pos','neu','comp']]
    y_train = train['confirmation_bias']
    X_test = test[['neg','pos','neu','comp']]
    y_test = test['confirmation_bias']

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("F1-score",metrics.f1_score(y_test, y_pred))

    export_prediction(y_test, y_pred)

    scores = cross_validation(final_df, clf)
    print("10-Fold Cross-Validation")
    print(scores)

    # Hyper Parameter Tuning - Grid Search
    param_grid = {'C': [50, 10, 1.0, 0.1, 0.01],
                  'gamma': ['scale'],
                  'kernel': ['poly', 'rbf', 'sigmoid']}

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0, refit = True, verbose = 3)
    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print(grid_result.best_params_)
    print(grid_result.best_estimator_)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions))
    
    pass


if __name__ == '__main__':
    main()
