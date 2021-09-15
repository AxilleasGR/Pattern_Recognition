import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import csv

def result(home_goals,away_goals):
    goal_diff = int(home_goals) - int(away_goals)
    if(goal_diff > 0):
        return "H"  #return letter H from Home winner
    elif(goal_diff == 0):
        return "D"  #return letter D from draw
    else:
        return "A"  #return letter A from Away winner



def observe(results,class_i,fold,each_fold,total_matches):
    y = []
    for i in range(0,fold*each_fold):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    for i in range((fold+1)*each_fold,total_matches):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    return y

def matrix(company):
    X = []
    for i in range(len(company)):   #company[i][0] stores the match_id.
        home_odd = company[i][1]    #company[i][1] stores odds for home win
        draw_odd = company[i][2]    #company[i][2] stores odds for draw
        away_odd = company[i][3]    #company[i][3] stores odds for away win
        X.append([home_odd,draw_odd,away_odd])
    X = np.array(X)
    X=np.insert(X,0,1.0,axis=1)
    return X


def k_fold_cross_validation(company,k_fold):
    matches = len(company)
    each_fold = int(matches/k_fold)
    num_of_matches = each_fold*k_fold
    training_set = []
    testing_set = []
    for fold in range(k_fold):
        start_test = fold*each_fold
        for m in range(0,num_of_matches,each_fold):
            if(m == start_test):
                f = company[m:m+each_fold]
                testing_set.append(company[m:m+each_fold])
            else:
                if(len(training_set) < fold+1):
                    training_set.append(company[m:m+each_fold])
                else:
                    training_set[fold] += company[m:m+each_fold]
    return training_set,testing_set


def score_weights(test_set,w,fold,match_results):
    outcome = ["H","D","A"]
    correct = 0
    wrong = 0
    step = len(test_set)
    start = fold*step
    stop = start+step
    for m in range(start,stop):
        result = []
        i = m-start
        for k in range(len(w)):
            y = w[k][0] + w[k][1]*test_set[i][1] + w[k][2]*test_set[i][2] + w[k][3]*test_set[i][3]
            result.append((1-y)**2)
        best_fit = result.index(min(result))
        if(outcome[best_fit] == match_results[m][1]):
            correct += 1
        else:
            wrong += 1
    return [correct,wrong]


def robbins_monro(X,y):
    def hypothesis(match,W):
        result = 0
        for i in range(len(W)):
            result += W[i]*X[match][i]
        return result
    init_weights = [0.0,0.0,0.0,0.0]
    weights = [0.0,0.0,0.0,0.0]
    num_of_matches = len(X)
    a = 1
    convergence = 0.0000001
    for match in range(1,num_of_matches):
        if(a <= convergence):
            break
        else:
            learning_rate = a/match #+0.001 bit more accurate results
            for w_i in range(4):
                J = y[match]-hypothesis(match,init_weights)
                step = learning_rate*J
                weights[w_i] = init_weights[w_i]+step
            init_weights = weights
    return weights
