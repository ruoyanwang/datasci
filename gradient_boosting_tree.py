from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from random import shuffle
import pandas as pd
import numpy
import math

train_path = '../data/train.csv'
test_path = '../data/test.csv'
sample_path = '../data/sampleSubmission.csv'
output_path = 'submission_gbt.csv'

def get_category_dict():
    global sample_path
    sample_path_f =  open(sample_path)
    first_line = sample_path_f.readline()
    first_line = first_line[:-1]
    
    category_dict = dict()
    cnt = 0
    for cat in first_line.split(',')[1:]:
        category_dict[cat] = cnt
        cnt += 1

    sample_path_f.close()
    return category_dict


def write_results(Y_test):
    global sample_path, output_path
    
    sample_path_f = open(sample_path)
    output_path_f = open(output_path, 'w')

    first_line = sample_path_f.readline()
    output_path_f.write(first_line)

    cnt = 0    
    for res in Y_test:
        line = str(cnt) 
        for i in range(len(first_line.split(',')) - 1):
            if res == i:
                line += (',' + str(1) )
            else:
                line += (',' + str(0) )
        line += '\n'
        output_path_f.write(line)
        cnt += 1

    sample_path_f.close()
    output_path_f.close()

def write_results_prob(Y_test):
    global sample_path, output_path
    
    sample_path_f = open(sample_path)
    output_path_f = open(output_path, 'w')

    first_line = sample_path_f.readline()
    output_path_f.write(first_line)

    cnt = 0    
    for res in Y_test:
        line = str(cnt) 
        for prob in res:
            line += (',' + str(prob) )

        line += '\n'
        output_path_f.write(line)
        cnt += 1

    sample_path_f.close()
    output_path_f.close()


category_dict = get_category_dict()
address_vector_dict = dict()

def get_feature(train_f, tag="training_set"):

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
    #train_f['X'] = (train_f['X'] - train_f['X'].min()) / (train_f['X'].max() - train_f['X'].min())
    #train_f['Y'] = (train_f['Y'] - train_f['Y'].min()) / (train_f['Y'].max() - train_f['Y'].min())



    ## Feature of 'Address'
    """
    # this one is better for random forests
    address_dict = dict()
    cnt = 0
    for address in train_f['Address']:
        if address not in address_dict:
            address_dict[address] = cnt
            cnt += 1
    # print district_dict
    
    train_f['Address'] = train_f['Address'].map(address_dict).astype(int)
    """

    if tag == "training_set":
        cnt = 0
        for address, cat in zip(train_f['Address'], train_f['Category']):
            if address not in address_vector_dict:
                address_vector_dict[address] = numpy.zeros((len(category_dict.keys())), dtype=int)
            address_vector_dict[address][category_dict[cat]] += 1

    train_f['Address'] = train_f['Address'].map(address_vector_dict)
    addr_feat = numpy.zeros((len(train_f['Address']), len(category_dict.keys()))     , dtype=int)
    
    cnt = 0
    for addr in train_f['Address']:
        addr_feat[cnt, :] = addr
        cnt += 1


    ## Features of 'Time'
    # print train_f['Dates'].dt.year
    """
    train_f['Time'] = train_f['Dates'].dt.hour*60 + train_f['Dates'].dt.minute
    train_f['Time'] = (train_f['Time'] - train_f['Time'].min()) / (train_f['Time'].max() - train_f['Time'].min())
    # print train_f['Time']
    """

    train_f['Year'] = train_f['Dates'].dt.year 
    train_f['Year'] = train_f['Year'] - train_f['Year'].min()

    train_f['Month'] = train_f['Dates'].dt.month 
    train_f['Month'] = train_f['Month'] - train_f['Month'].min()
    
    train_f['Hour'] = train_f['Dates'].dt.hour
    train_f['Hour'] = train_f['Hour'] - train_f['Hour'].min()

        
    if tag == "training_set":

        # category_dict = get_category_dict()
        train_f['Category'] = train_f['Category'].map(category_dict).astype(int)
        # print train_f['Category']
        
        # Transform data to numpy matrix
        Y = train_f['Category'].values



    elif tag == "test_set":
        Y = None

    else:
        raise ValueError("tag must be training_set or test_set")

    # X = train_f[['DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y', 'Year', 'Month', 'Hour']].values  

    # X = numpy.concatenate((train_f[['DayOfWeek', 'PdDistrict', 'X', 'Y', 'Year', 'Month', 'Hour']].values , addr_feat), axis=1)

    X = numpy.concatenate((train_f[['X', 'Y', 'Year', 'Month', 'Hour']].values , addr_feat), axis=1)
    X = numpy.concatenate((X, pd.get_dummies(train_f['PdDistrict']).values), axis=1)
    X = numpy.concatenate((X, pd.get_dummies(train_f['DayOfWeek']).values), axis=1)

    return X, Y



def shuffle_XY(X, Y):
    assert len(X) == len(Y)
    data_len = len(X)

    index = range(data_len)
    shuffle(index)
    return X[index], Y[index]



def main():
    
    train_f = pd.read_csv(train_path, header=0, parse_dates=['Dates'])
    print train_f.dtypes

    X, Y = get_feature(train_f, "training_set")
    

    ### TRAINING
    clf = GradientBoostingClassifier(n_estimators=50)
    # clf = RandomForestClassifier(n_estimators=2)
    # clf = LogisticRegression(n_jobs=4)

    X, Y = shuffle_XY(X, Y)
    data_len = len(X)
    train_len = data_len * 95 / 100 
    val_len = data_len - train_len
    X_train = X[:train_len]
    X_val = X[train_len:]
    Y_train = Y[:train_len]
    Y_val = Y[train_len:]
    
    clf = clf.fit(X_train, Y_train)
    print "Training done"

    
    val_acc = clf.score(X_val, Y_val)
    print "Val acc:", val_acc

    val_pred = clf.predict_proba(X_val)
    

    # print max(Y_val), min(Y_val)
    # print Y_val, Y_val + 1
    val_log = 0.0
    cnt = 0
    for y in Y_val:
        val_log += math.log(val_pred[cnt, y]+0.0000001)
        cnt += 1
    val_log =  - val_log / len(Y_val)
    print "Val log loss:", val_log
 
    # print "Val loss:", log_loss(Y_val+1, val_pred) # Note the +1 here!
    """
    # scores = cross_val_score(clf, X, Y)
    # print "Cross val acc:", scores.mean()
    """

    ### Testing

    test_f = pd.read_csv(test_path, header=0, parse_dates=['Dates'])
    # print test_f.dtypes

    X_test, _ = get_feature(test_f, "test_set")
    Y_test = clf.predict_proba(X_test)

    ### Write results
    # write_results(Y_test)
    write_results_prob(Y_test)

    
if __name__ == "__main__":
    main()

