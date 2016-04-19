from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from random import shuffle
import pandas as pd
import numpy

train_path = '../data/train.csv'
test_path = '../data/test.csv'

def main():
    
    train_f = pd.read_csv(train_path, header=0, parse_dates=['Dates'])
    print train_f.dtypes

    ## Features of 'DayOfWeek'
    train_f['DayOfWeek'] = train_f['DayOfWeek'].map( \
    {'Monday':0,
     'Tuesday':1,
    'Wednesday':2,
    'Thursday':3,
    'Friday':4,
    'Saturday':5,
    'Sunday':6} \
    ).astype(int)
    # print train_f['DayOfWeek']

    ## Features of 'PdDistrict'
    district_dict = dict()
    cnt = 0
    for district in train_f['PdDistrict']:
        if district not in district_dict:
            district_dict[district] = cnt
            cnt += 1
    # print district_dict
    
    train_f['PdDistrict'] = train_f['PdDistrict'].map(district_dict).astype(int)
    # print train_f['PdDistrict']
    
    ## Features of 'X' and 'Y'
    train_f['X'] = (train_f['X'] - train_f['X'].min()) / (train_f['X'].max() - train_f['X'].min())
    
    train_f['Y'] = (train_f['Y'] - train_f['Y'].min()) / (train_f['Y'].max() - train_f['Y'].min())


    ## Features of 'Time'
    # print train_f['Dates'].dt.time
    train_f['Time'] = train_f['Dates'].dt.hour*60 + train_f['Dates'].dt.minute
    train_f['Time'] = (train_f['Time'] - train_f['Time'].min()) / (train_f['Time'].max() - train_f['Time'].min())
    # print train_f['Time']
        

    ## Labels
    category_dict = dict()
    cnt = 0
    for category in train_f['Category']:
        if category not in category_dict:
            category_dict[category] = cnt
            cnt += 1
    # print category_dict
    
    train_f['Category'] = train_f['Category'].map(category_dict).astype(int)
    # print train_f['Category']


    # Transform data to numpy matrix
    X = train_f[['DayOfWeek', 'PdDistrict', 'X', 'Y', 'Time']].values
    Y = train_f['Category'].values

    
    ### TRAINING
    clf = RandomForestClassifier(n_estimators=200)
    # clf = LogisticRegression(n_jobs=4)

    X, Y = shuffle_XY(X, Y)
    data_len = len(X)
    train_len = data_len * 9 / 10
    val_len = data_len - train_len
    X_train = X[:train_len]
    X_val = X[train_len:]
    Y_train = Y[:train_len]
    Y_val = Y[train_len:]
    
    clf = clf.fit(X_train, Y_train)
    print "Training done"
    
    train_acc = clf.score(X_train, Y_train)
    print "Train acc:", train_acc
    
    val_acc = clf.score(X_val, Y_val)
    print "Val acc:", val_acc
    
    # scores = cross_val_score(clf, X, Y)
    # print "Cross val acc:", scores.mean()



def shuffle_XY(X, Y):
    assert len(X) == len(Y)
    data_len = len(X)

    index = range(data_len)
    shuffle(index)
    return X[index], Y[index]

    
if __name__ == "__main__":
    main()

