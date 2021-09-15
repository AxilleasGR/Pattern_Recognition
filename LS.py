#import oti xriazete
import numpy
import matplotlib.pyplot as plotter

plotter.style.use('fivethirtyeight') #style sti matplotlib
from mpl_toolkits.mplot3d import Axes3D

import matfunct.functions as extfunc

#orismos ton dedomenon
Match_List = []
Match_Results_List = []

#apodosis
companies_odds = [[]for x in range(4)]

#lista apo betting companies
companies = ["B365","BW","IW","LB"]
#anigma matches table 
with open("./csv_db/match.csv","r") as match_csv:
    for line in match_csv:
        Match_List.append(line[:-1].split(","))
#diagrapse to proto entry to opoio einai ta column names
del Match_List[0]

#plithos ton matches
matches_length = len(Match_List)

#arithmos eterion
companies_length = len(companies_odds)

#epanelave (iteration) gia kathe match
for match in Match_List:
    goals_H = match[9]   #Posa goals evale i edos edras omada
    goals_A = match[10]  #Posa goals evale i ektos edras
    result = extfunc.result(goals_H,goals_A)    #Vlepoume to result san H, D, A analoga me ti diafora sto score
    Match_Results_List.append([int(match[6]),result])            #i lista Match_Results_List kratai ta match results gia kathe match_id. Match_Results_List[match,result]

#Epanelave gia kathe eteria kai gia kathe match....
index = 11
for company in range(companies_length):
    for match in range(matches_length):
        companies_odds[company].append([int(Match_List[match][6]),float(Match_List[match][index]), float(Match_List[match][index+1]), float(Match_List[match][index+2])])
    index += 3
#Ipologismos varititas
k = 10 #10 fold cross validation

company_fold_weight = [[]for x in range(companies_length)]

#training set
training_set = [[]for x in range(companies_length)]

#testing set
testing_set = [[]for x in range(companies_length)]

#elenxos gia kathe eteria
for company in range(companies_length):
    training_set_i,testing_set_i = extfunc.k_fold_cross_validation(companies_odds[company],k)
    training_set_matches = len(training_set_i[company])
    testing_set_matches = len(testing_set_i[company])
    matches_length = training_set_matches + testing_set_matches
    #for-each eteria
    for fold in range(k):
        X = extfunc.matrix(training_set_i[fold])
        fold_weights = [[]for x in range(3)]
        for i,class_i in enumerate(["H","D","A"]):
            y = extfunc.observe(Match_Results_List,class_i,fold,testing_set_matches,matches_length)
            w_i = numpy.linalg.inv(X.T @ X) @ (X.T @ y)
            fold_weights[i] = w_i
        company_fold_weight[company].append((fold_weights))
    training_set[company] = (training_set_i)
    testing_set[company] = (testing_set_i)

#elenxos varititas (weight testing) kai evaluation

scores = [[]for x in range(companies_length)]


Weight = []
#Ta results apo to score weights gia kathe eteria
for company in range(companies_length):
    for fold in range(k):
        results = extfunc.score_weights(testing_set[company][fold],company_fold_weight[company][fold],fold,Match_Results_List)
        scores[company].append(results)

#to kalitero dinato fold
for company,company_score in enumerate(scores):  #edo ginete tuple unpacking
    max_score = company_score[0][0]
    best_fold = 0
    for fold in range(1,k):
        if(company_score[fold][0] > max_score):
            max_score = company_score[fold][0]
            best_fold = fold

    best_score = (max_score/testing_set_matches)*100                    #Score tou best fold san pososto correct/total_match*100
    
    
    Weight.append(company_fold_weight[company][best_fold])              #Appending/apothekfsi weights ton kaliteron fold ana company
    
    scores[company] = [best_fold,int(best_score)]



# Kane to plotting kai ektiposi stin othoni
xx, yy = numpy.meshgrid(range(10), range(10)) #rithmisis grid
names = ["HOME","DRAW","AWAY"]
colors = ["green","red","purple"]

for c,company in enumerate(Weight): #tuple unpacking on enumerated Weights
    figure = plotter.figure()
    axes = figure.add_subplot(111, projection='3d')
    print("Best weights for company: ",companies[c])
    for j,plane in enumerate(company):
        print("FOR ",names[j]," - ",plane)
        z = (-plane[1]*xx - plane[2]*yy - plane[0])/plane[3]
        axes.plot_surface(xx, yy, z, color = colors[j],alpha = 0.5)
        
        
    print("\n\n") # newline separator

    #theloume scatter plot
    to_scatter_home_wins = []
    to_scatter_draw = []
    to_scatter_away_wins = []

    for i in range(matches_length):
        if Match_Results_List[i][1] == "H": #an einai HOME
            to_scatter_home_wins.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])
        elif Match_Results_List[i][1] == "D": #an einai DRAW
            to_scatter_draw.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])
        else: #diaforetika, diladi an einai AWAY
            to_scatter_away_wins.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])

    #metetrepse se numpy array
    to_scatter_home_wins = numpy.array(to_scatter_home_wins)
    to_scatter_draw = numpy.array(to_scatter_draw)
    to_scatter_away_wins = numpy.array(to_scatter_away_wins)


    
    
    #kane to plot (scatter)
    axes.scatter(to_scatter_home_wins[:,0],to_scatter_home_wins[:,1],to_scatter_home_wins[:,2],color = colors[0],label = "nikes (entos)")
    axes.scatter(to_scatter_draw[:,0],to_scatter_draw[:,1],to_scatter_draw[:,2],color = colors[1], label = "isopalia")
    axes.scatter(to_scatter_away_wins[:,0],to_scatter_away_wins[:,1],to_scatter_away_wins[:,2],color = colors[2], label = "nikes (ektos)")

    #vale titlo sto grafima
    plotter.title("apodosis kai veltista oria apofaseon - "+companies[c]+
    "\n score (pososto axiologisis): "+str(scores[c][1])+"%, me fold - "+str(scores[c][0])+"/"+str(k))
    axes.set_xlabel("nikes (entos)") #x axis label
    axes.set_ylabel("isopalia") #y axis label
    axes.set_zlabel("nikes (ektos)") #z axis label
    plotter.legend()
    plotter.tight_layout() #diorthose to formatting/layout

#emfanise to plot stin othoni
plotter.show()
