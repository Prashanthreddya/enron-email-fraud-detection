#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data






### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list=['poi','salary' , 'deferral_payments', 'total_payments', 'loan_advances','bonus', 'restricted_stock_deferred', 
               'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
               'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
               'from_this_person_to_poi', 'shared_receipt_with_poi']
temp_list=features_list
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers

def PlotOutlier(data_dict, x_label, y_label):
    #poi's are plotted in red and non-poi's in blue
    data = featureFormat(data_dict, [x_label, y_label, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#print(PlotOutlier(data_dict, 'salary', 'bonus'))	
###the above scatter shows that TOTAL is an outlier
data_dict.pop('TOTAL')
#print(PlotOutlier(data_dict, 'salary', 'bonus'))			
outlier_count=1

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict
###here we create new features nnamely : percentage_to_poi, bonus_to_salary_ratio 

def percentage_poi(poi_only,total):
	if poi_only=='NaN' or total=='NaN':
		#i.e if either the mail count of poi(from or to) is NaN or the total mail count is NaN we return 0
		return 0
	else:
		return poi_only/total

def compute_bon_sal_rat(bonus,salary):
	if bonus=='NaN' or salary=='NaN':
		return 0
	else:
		return bonus/salary



##we now compute the two new features i.e percentage_to_poi, bonus_to_salary_ratio for each entry in the dataset
##we use the copied dictionary that is my_dataset
for entry in my_dataset:
	###call percentage_poi twice, once for creating percentage_from poi and other for percentage_to_poi
    point=my_dataset[entry]
    ptp=percentage_poi(point["from_this_person_to_poi"],point["from_messages"])
    point["percentage_to_poi"]=ptp


    btsr=compute_bon_sal_rat(point["bonus"],point["salary"])
    point["bonus_to_salary_ratio"]=btsr


features_list=features_list+["percentage_to_poi","bonus_to_salary_ratio"]


### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)







##feature selection phase using selectKbest algorithm

from sklearn.feature_selection import SelectKBest

def extract_best_features(data_dict, curt_features_list, num):
	##using SelectKBest function of sklearn we chose features based on the importance

    k_best = SelectKBest(k=num)
    k_best.fit(features, labels)
    scores = k_best.scores_
    ##we have to combine the feature list and scores list into a list of tuples using zip function
    pairs=zip(features_list[1:], scores)
    ##we sort the above list of tuples over the scores(i.e the second value in each tuple)
    pairs=sorted(pairs,key=lambda x:x[1])
    a=len(pairs)-1
    best_features=[]
    print "\n\n**********Features and there importance**********\n"
    for i in range(0,num):
    	best_features.append(pairs[a][0])
        print '{:>30} ->  {:>20}'.format(pairs[a][0],pairs[a][1])
    	a=a-1
    return best_features

#note : we can print the feature list before and after calling the function to check the changes 
###we try different values for K and chose the optimal one. 
best_features=extract_best_features(my_dataset, features_list,8)

features_list=["poi"]+best_features


##we now determine the salient features of the dataset
#1. total number of data-points
print "\nTotal number of Data Points (including outlier): ",len(data_dict)+outlier_count
#2. total  number of outliers, here after we will not be considering this outlier as a part of the dataset
print "Total number of outliers: ",outlier_count
#3. allocation across classes 
num_poi=0
for p in data_dict.values():
    if p['poi']:
        num_poi += 1
print "Total number of Person of Interest : ",num_poi
print "Total number of Non-Person of Interest : ",len(data_dict)-num_poi
print "Total number of features used : ",len(features_list)

#4. to determine features in features_list with more NaN values
NaNdict=dict()
for i in temp_list:
    NaNdict[i]=0

for person,feature in data_dict.iteritems():
    for f in temp_list:
        if feature[f]=='NaN':
            NaNdict[f]+=1
print "\n\n**********Features and NaN entry Count**********\n"
for feature in NaNdict:
    if NaNdict[feature]>=5:
        print '{:>30} ->  {:>20}'.format(feature,NaNdict[feature])


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



##naive bayes classifier
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

##decision tree classfier
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(min_samples_split=40)

##kmeans clustering classifier
from sklearn.cluster import KMeans
km_clf = KMeans(n_clusters=2,tol=0.001)

##adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()

##random forest classifier
from sklearn.ensemble import RandomForestClassifier
ran_clf = RandomForestClassifier()

##principle component analysis
from sklearn.decomposition import PCA
pca=PCA()

##scaler for scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


## in the pipeline below, we test it with various possible clf's. the optimal one is assigned at the end (Naive Bayes in this case)
## the pipeline here is made up of 3 steps : MinMaxScaker, PCA and the classifier. 

from sklearn.pipeline import Pipeline
#for adaboost replace nb_clf with ada_clf and remove pca from pipeline
#for kmeans replace nb_clf with km_clf, retain pca
#for random forest replace nb_clf with ran_clf, retain pca
pipeline = Pipeline([('scale',scaler),('pca',pca),('clf',nb_clf)])




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)




###Naive bayes and Adaboost gave me better results in comparison with others.
###I found naive bayes optimal over adaboost, as the results of naive bayes were better than adaboost

###tuning adaboost classifier

'''
cross_strat = StratifiedShuffleSplit(
    labels_train,
    test_size = 0.5,
    random_state = 42
)

ada_parameters = {'clf__n_estimators': [50,100,150,200], 'clf__learning_rate': [0.4,0.6,0.8,1.0],'clf__algorithm':['SAMME', 'SAMME.R']}

grid_res = GridSearchCV(estimator=pipeline, param_grid=ada_parameters, scoring="recall", cv=cross_strat, error_score=0)
grid_res.fit(features_train, labels_train)
labels_predictions = grid_res.predict(features_test)

clf = grid_res.best_estimator_
##note : remove pca in the pipeline step and change the classifier from nb_clf to ada_clf
'''

#algorithm performances
'''  
Naive Bayes     : Accuracy -> 0.83907; Precision -> 0.42841; Recall -> 0.37850;
Adaboost        : Accuracy -> 0.83700; Precision -> 0.32819; Recall -> 0.21250;
Kmeans          : Accuracy -> 0.70620; Precision -> 0.18700; Recall -> 0.35950;
Random forest   : Accuracy -> 0.81300; Precision -> 0.29008; Recall -> 0.21350;
'''


###assigning pipeline with naive bayes classifier to the classifier variable clf
clf=pipeline

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
