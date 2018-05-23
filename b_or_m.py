import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df= pd.read_csv('wdbc.data.txt', header= 0)

col_names=  ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 
df.columns= col_names

df.drop(['id_number'], 1, inplace= True)
'''
sns.countplot(df['diagnosis'], label="Class Count")
plt.show()
'''
df['diagnosis']= df['diagnosis'].map({'M': 1, 'B': 0})
#print(df['diagnosis'].value_counts())
'''
list_= ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se']
df_se= df.loc[:, list_]
corr= df_se.corr()
corr= pd.DataFrame(corr)
df_se.columns= list_
sns.heatmap(corr, annot= True, cbar= True, cmap= 'YlOrBr')
plt.show()
list__= ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean']
df_mean= df.loc[:, list__]
corr= df_mean.corr()
corr= pd.DataFrame(corr)
df_mean.columns=list__
sns.heatmap(corr, annot= True, cbar= True, cmap= 'YlOrBr')
plt.show()
list___= ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst']
df_worst= df.loc[:, list___]
corr= df_worst.corr()
corr= pd.DataFrame(corr)
df_worst.columns=list___
sns.heatmap(corr, annot= True, cbar= True, cmap= 'YlOrBr')
plt.show()

co_= pd.DataFrame(df.corr())
co_.columns= col_names[1:]
co_= co_.loc[:, ['diagnosis']]
co_= co_.abs()
co_= co_.unstack()
so_= co_.sort_values(kind= 'quicksort')
print(so_)
'''

'''
c = df_mean.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
print(so.tail(30))
'''
'''
Highly correlated columns:
compactness_se     concave_points_se    0.743973
concave_points_se  compactness_se       0.743973
                   concavity_se         0.771614
concavity_se       concave_points_se    0.771614
compactness_se     concavity_se         0.801183
concavity_se       compactness_se       0.801183
area_se            perimeter_se         0.936973
perimeter_se       area_se              0.936973
area_se            radius_se            0.951299
radius_se          area_se              0.951299
perimeter_se       radius_se            0.972555
radius_se          perimeter_se         0.972555

radius_se, perimeter_se, area_se-> radius_se
concavity_se, concave_points_se, compactness_se-> concave_points_se


radius_mean          concave_points_mean    0.823165
concave_points_mean  radius_mean            0.823165
area_mean            concave_points_mean    0.824246
concave_points_mean  area_mean              0.824246
compactness_mean     concave_points_mean    0.829050
concave_points_mean  compactness_mean       0.829050
perimeter_mean       concave_points_mean    0.851338
concave_points_mean  perimeter_mean         0.851338
compactness_mean     concavity_mean         0.881619
concavity_mean       compactness_mean       0.881619
concave_points_mean  concavity_mean         0.920462
concavity_mean       concave_points_mean    0.920462
area_mean            perimeter_mean         0.986548
perimeter_mean       area_mean              0.986548
area_mean            radius_mean            0.987344
radius_mean          area_mean              0.987344
perimeter_mean       radius_mean            0.997876
radius_mean          perimeter_mean         0.997876

radius_mean, area_mean, perimeter_mean-> radius_mean
concavity_mean, concave_points_mean, compactness_mean-> concave_points_mean


area_worst            concave_points_worst    0.745417
concave_points_worst  area_worst              0.745417
                      radius_worst            0.785908
radius_worst          concave_points_worst    0.785908
compactness_worst     concave_points_worst    0.799028
concave_points_worst  compactness_worst       0.799028
perimeter_worst       concave_points_worst    0.814596
concave_points_worst  perimeter_worst         0.814596
                      concavity_worst         0.854217
concavity_worst       concave_points_worst    0.854217
compactness_worst     concavity_worst         0.891409
concavity_worst       compactness_worst       0.891409
area_worst            perimeter_worst         0.977475
perimeter_worst       area_worst              0.977475
area_worst            radius_worst            0.983919
radius_worst          area_worst              0.983919
perimeter_worst       radius_worst            0.993814
radius_worst          perimeter_worst         0.993814

concavity_worst, concave_points_worst, compactness_worst-> concave_points_worst
area_worst, perimeter_worst, radius_worst-> radius_worst
'''

col_names_1_mod=  ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst']
'''
for i in col_names_1_mod:
    sns.set()
    sns.violinplot(x= 'diagnosis', y= i, data= df)
    plt.show()
'''
'''
Good classifiers:
    1. radius_mean
    2. area_mean
    3. perimeter_mean
    4. concavity_mean
    5. concave_points_mean
    6. radius_se
    7. perimeter_se
    8. area_se
    9. concave_points_se
    10. radius_worst
    11. perimeter_worst
    12. area_worst
    13. concavity_worst
    14. concave_points_worst
'''
'''
for i in col_names_1_mod:
    sns.set()
    sns.stripplot(x= 'diagnosis', y= i, jitter= True, data= df)
    plt.show()
'''
'''
plt.figure(figsize= (25, 25))
plt.subplot(331)
sns.swarmplot(x= 'diagnosis', y= 'radius_mean', data= df)
plt.subplot(332)
sns.swarmplot(x= 'diagnosis', y= 'area_mean', data= df)
plt.subplot(333)
sns.swarmplot(x= 'diagnosis', y= 'perimeter_mean', data= df)
plt.subplot(334)
sns.swarmplot(x= 'diagnosis', y= 'radius_se', data= df)
plt.subplot(335)
sns.swarmplot(x= 'diagnosis', y= 'area_se', data= df)
plt.subplot(336)
sns.swarmplot(x= 'diagnosis', y= 'perimeter_se', data= df)
plt.subplot(337)
sns.swarmplot(x= 'diagnosis', y= 'radius_worst', data= df)
plt.subplot(338)
sns.swarmplot(x= 'diagnosis', y= 'area_worst', data= df)
plt.subplot(339)
sns.swarmplot(x= 'diagnosis', y= 'perimeter_worst', data= df)
plt.show()
'''
'''
plt.figure(figsize= (25, 25))
plt.subplot(331)
sns.swarmplot(x= 'diagnosis', y= 'concavity_mean', data= df)
plt.subplot(332)
sns.swarmplot(x= 'diagnosis', y= 'concave_points_mean', data= df)
plt.subplot(333)
sns.swarmplot(x= 'diagnosis', y= 'compactness_mean', data= df)
plt.subplot(334)
sns.swarmplot(x= 'diagnosis', y= 'concavity_se', data= df)
plt.subplot(335)
sns.swarmplot(x= 'diagnosis', y= 'concave_points_se', data= df)
plt.subplot(336)
sns.swarmplot(x= 'diagnosis', y= 'compactness_se', data= df)
plt.subplot(337)
sns.swarmplot(x= 'diagnosis', y= 'concavity_worst', data= df)
plt.subplot(338)
sns.swarmplot(x= 'diagnosis', y= 'concave_points_worst', data= df)
plt.subplot(339)
sns.swarmplot(x= 'diagnosis', y= 'compactness_worst', data= df)
plt.show()
'''
'''
Better Classifiers:
    1. Out of radius_mean, perimeter_mean and area_mean-> area_mean
    2. Out of radius_se, perimeter_se and area_se-> area_se
    3. Out of radius_worst, perimeter_worst and area_worst-> area_worst
    4. Out of concavity_mean, concave_points_mean and compactness_mean-> concavity_mean
    5. Out of concavity_se, concave_points_se and compactness_se-> concavity_se
    6. Out of concavity_worst, concave_points_worst and compactness_worst-> concavity_worst
    7. fractal dimension (s)-> do not matter to the classification
    8. symmetry dimension(s)-> do not matter to the classification
    9. texture dimension (s)-> do not matter to the classification
    10. smoothness dimension(s)-> do not matter to the classification
'''
'''
col_names_1_mod=  ['area_mean', 'concavity_mean',
         'area_se', 'concavity_se',
         'area_worst', 'concavity_worst']

col_namess_2_mod1= ['fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst',
                    'symmetry_mean', 'symmetry_se', 'symmetry_mean', 
                    'smoothness_mean', 'smoothness_se', 'smoothness_worst',
                    'texture_mean', 'texture_se', 'texture_worst', 
                    'compactness_mean', 'compactness_se', 'compactness_worst']

doubtful_cols_mod1= ['texture_worst', 'compactness_mean', 'compactness_worst']

includ_doubtful_full_cols= ['area_mean', 'concavity_mean',
         'area_se', 'concavity_se',
         'area_worst', 'concavity_worst', 'texture_worst']
'''
includ_doubtful_full_cols= ['radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean','symmetry_mean',
       'radius_se', 'texture_se', 'smoothness_se','concavity_se', 'symmetry_se','radius_worst', 'texture_worst',
       'smoothness_worst','concavity_worst','symmetry_worst']

'''
#From RFE:
includ_doubtful_full_cols= ['radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean', 'radius_se', 'radius_worst', 'smoothness_worst', 'concavity_worst', 'symmetry_worst']
'''
df_idfc= df.loc[:,includ_doubtful_full_cols]
'''
scaler= StandardScaler()
scaled_df_idfc= pd.DataFrame(scaler.fit_transform(df_idfc), columns= includ_doubtful_full_cols)
for _ in includ_doubtful_full_cols:
    plt.title('Before Scaling')
    sns.kdeplot(df_idfc[_])
    plt.show()
    plt.title('After Scaling')
    sns.kdeplot(scaled_df_idfc[_])
    plt.show()
'''

'''
Plotting doubtful_cols_mod1:
g = sns.PairGrid(df,
                 x_vars='diagnosis',
                 y_vars=doubtful_cols_mod1,
                 aspect=.75, size=7.0)
g.map(sns.violinplot, palette="pastel")
plt.show()
'''
X= np.array(df_idfc)
y= np.array(df['diagnosis'])

train_arr= []
test_arr= []
kf= KFold(len(X), shuffle= True, random_state= 4)
for train, test in kf:
    train_arr.append(train)
    test_arr.append(test)
x_train, x_test, y_train, y_test= X[train], X[test], y[train], y[test]
'''
cross_vals= []
for i in range(1, 50):
    if(i%2 != 0):
        knn= KNeighborsClassifier(n_neighbors= i)
        scores= cross_val_score(knn, X, y, cv=10, scoring= 'accuracy')
        cross_vals.append(scores.mean())
cross_vals_e= list(1 - np.array(cross_vals))
k_list= [i for i in range(1, 50) if(i%2)!=0]
optimal_k= k_list[cross_vals_e.index(min(cross_vals_e))]
print(optimal_k, min(cross_vals_e))
plt.scatter(k_list, cross_vals_e, c= 'g', s= 100)
plt.plot(k_list, cross_vals_e, c= 'k')

knn= KNeighborsClassifier()
tuned_params= [{'n_neighbors': [11, 13, 15, 17, 19, 21], 'algorithm': ['ball_tree', 'kd_tree'], 'weights': ['uniform', 'distance']}]
lrgs = GridSearchCV(estimator=knn, param_grid= tuned_params, n_jobs=1, cv= 10)
print(np.mean([lrgs.fit(x_train, y_train).score(x_test,y_test) for train, test in kf]))
print(pd.DataFrame(lrgs.grid_scores_).head(20))
print(lrgs.best_score_)
print(lrgs.best_estimator_.n_neighbors)
print(lrgs.best_estimator_.algorithm)
print(lrgs.best_estimator_.weights)
print(lrgs.best_estimator_.leaf_size)
'''
'''
knn= KNeighborsClassifier(n_neighbors= 13, algorithm= 'ball_tree', weights= 'uniform', leaf_size= 30)
knn= knn.fit(x_train, y_train)
y_pred= knn.predict(x_test)

cm= pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
print(cm)
cm_= cm.apply(lambda r: 100.0*r/r.sum())
print(cm_)
sns.set(font_scale= 1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt= '')
plt.show()

cv_scores = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)
avg_score = np.mean(cv_scores)
sd_score = np.std(cv_scores)
print('avg_score:', avg_score, 'sd_score:', sd_score)

y_pred= knn.predict(x_test)
y_pred_prob= knn.predict_proba(x_test)[:, 1].ravel()
print(np.mean([knn.fit(X[train],y[train]).score(X[test],y[test]) for train, test in kf]))
'''
'''
fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)
roc_auc= auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', label= roc_auc)
plt.legend(loc='lower right')
plt.plot(fpr, tpr, label= 'KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
gini= (2*roc_auc)-1
print(gini)
'''

'''
Repeating over a set of random_states:
    criterion= entropy
    max_features= sqrt
    max_depth= 38
    min_samples_split= 4 
    min_samples_leaf= 9
    or
    criterion= entropy
    max_features= sqrt
    max_depth= 8
    min_samples_split= 10 
    min_samples_leaf= 1
    or
    criterion= entropy
    
'''
'''
dt= DecisionTreeClassifier(criterion= 'entropy', max_features= 'sqrt', max_depth= 9, min_samples_split=12, min_samples_leaf= 6, random_state= 123)
dt= dt.fit(x_train, y_train)
f_i_list= dt.feature_importances_
ind= np.argsort(f_i_list)
plt.title('Feature Importances')
plt.barh(range(len(ind)), f_i_list[ind], color='b', align='center')
plt.yticks(range(len(ind)), includ_doubtful_full_cols)
plt.show()
y_pred= dt.predict(x_test)
cm= pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
print(cm)
cm_= cm.apply(lambda r: 100.0*r/r.sum())
print(cm_)
sns.set(font_scale= 1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt= '')
plt.show()
cv_scores = cross_val_score(estimator=dt, X=x_train, y=y_train, cv=10)
avg_score = np.mean(cv_scores)
sd_score = np.std(cv_scores)
print('avg_score:', avg_score, 'sd_score:', sd_score)
y_pred_prob= dt.predict_proba(x_test)[:, 1].ravel()
print(np.mean([dt.fit(X[train],y[train]).score(X[test],y[test]) for train, test in kf]))
fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)
roc_auc= auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', label= roc_auc)
plt.legend(loc='lower right')
plt.plot(fpr, tpr, label= 'DT')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
gini= (2*roc_auc)-1
print(gini)
'''
'''
Predicted    0   1
True              
0          112   2
1           11  64
'''
'''
rfc= RandomForestClassifier(random_state= 122, n_estimators= 20, max_depth= 3, min_samples_split= 2)
rfc= rfc.fit(x_train, y_train)
rfe= RFE(rfc, 5)
rfe= rfe.fit(x_train, y_train)
print(list(rfe.support_))
print(list(rfe.ranking_))
plt.xticks(range(len(rfe.ranking_)), includ_doubtful_full_cols, rotation= 90)
plt.scatter(range(len(rfe.ranking_)), rfe.ranking_)
#From RFE-> 'radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean', 'radius_se', 'radius_worst', 'smoothness_worst', 'concavity_worst', 'symmetry_worst'

'''
'''
rfc= RandomForestClassifier(random_state= 122, n_estimators= 20, max_depth= 3, min_samples_split= 2)
rfc= rfc.fit(x_train, y_train)
f_i_list= rfc.feature_importances_
ind= np.argsort(f_i_list)
plt.title('Feature Importances')
plt.barh(range(len(ind)), f_i_list[ind], color='b', align='center')
plt.yticks(range(len(ind)), includ_doubtful_full_cols)
plt.show()
'''
'''
RandomForestClassifier(random_state= 122, n_estimators= 20, max_depth= 3, min_samples_split= 2)
False    0   1
True          
0      112   2
1        3  72

rfc= RandomForestClassifier(random_state= 122, n_estimators= 24, max_depth= 9, min_samples_split= 2)

False    0   1
True          
0      110   4
1        3  72
'''
'''
y_pred= rfc.predict(x_test)
cm= pd.crosstab(y_test, y_pred, rownames= ['True'], colnames= ['False'])
print(cm)
cm_= cm.apply(lambda r: 100.0*r/r.sum())
print(cm_)
sns.heatmap(cm, annot= True, annot_kws= {"size": 16}, fmt= '')
plt.show()
cv_scores= cross_val_score(estimator= rfc, X= x_train, y= y_train, cv= 10)
avg_score= np.mean(cv_scores)
std_score= np.std(cv_scores)
print('avg-score:', avg_score, '\nstd-score', std_score)
print('Classification Report')
print(classification_report(y_test,y_pred))
y_pred_prob= rfc.predict_proba(x_test)[:, 1]
print(np.mean(rfc.score(x_test, y_test)))
fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)
roc_auc= auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', label= roc_auc)
plt.legend(loc= 'lower right')
plt.plot(fpr, tpr, label= 'RFC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
gini= (2*roc_auc)-1
print(gini)
#Optimal value of k is 19.
#Excluding texture_worst-> 23
#Check for the most optimal classification pair.
'''
'''
Predicted    0   1
True              
0          110   4
1            9  66

Excluding texture_worst:
Predicted    0   1
True              
0          111   3
1            9  66
'''

'''
score-> 0.923482849604
n_neighbors-> 13
algorithm-> ball_tree
weights-> uniform
leaf_size-> 30
'''
'''
Exluding texture_worst-> 
0.934036939314
15
ball_tree
uniform
30
'''
'''
svc= SVC(gamma= 0.00001, C= 1000, kernel= 'linear', probability= True)
False    0   1
True          
0      111   3
1        6  69
'''
'''
svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
print(svc.fit(x_train, y_train).score(x_test, y_test))
svc= SVC(gamma= 0.00001, C= 1000, kernel= 'linear', probability= True)
print(svc.fit(x_train, y_train).score(x_test, y_test))
'''
print("Using the KMeans Clustering technique:\n")

feat_cols_sm = ['radius_mean', 'texture_mean', 'smoothness_mean']

# Use Pandas dataframe query to populate Numpy array with feature vectors.
X= np.array(df[feat_cols_sm])

# Initialize the KMeans cluster module. Setting it to find two clusters, hoping to find malignant vs benign.
clusters = KMeans(n_clusters=2, max_iter=10)

# Fit model to our selected features.
clusters.fit(X)

# Put centroids and results into variables.
centroids = clusters.cluster_centers_
labels = clusters.labels_

# Sanity check
print(centroids)

# Create new MatPlotLib figure
fig = plt.figure()
# Add 3rd dimension to figure
ax = fig.add_subplot(111, projection='3d')
# This means "red" and "blue"
colors = ["r", "b"]

# Plot all the features and assign color based on cluster identity label
for i in range(len(X)):
    ax.scatter(xs=X[i][0], ys=X[i][1], zs=X[i][2],
               c=colors[labels[i]], zdir='z')

# Plot centroids, though you can't really see them.
ax.scatter(xs=centroids[:,0], ys=centroids[:,1], zs=centroids[:,2],
           marker="x", s=150, c="c")

# Create array of diagnosis data, which should be same length as labels.
diag = np.array(y)
# Create variable to hold matches in order to get percentage accuracy.
matches = 0

# Transform diagnosis vector from B||M to 0||1 and matches++ if correct.
for i in range(0, len(diag)):
    if diag[i] == "B":
        diag[i] = 0
    if diag[i] == "M":
        diag[i] = 1
    if diag[i] == labels[i]:
        matches = matches + 1

#Calculate percentage matches and print.
percentMatch = (matches/len(diag))*100
print("Percent matched between benign and malignant ", percentMatch)

#Set labels on figure and show 3D scatter plot to visualize data and clusters.
ax.set_xlabel("Radius Mean")
ax.set_ylabel("Texture Mean")
ax.set_zlabel("Symmetry Mean")
plt.show()

#Finish