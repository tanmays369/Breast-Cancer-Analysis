import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import style
from sklearn.preprocessing import normalize

df= pd.read_csv('wdbc.data.txt', header= 0)
style.use('fivethirtyeight')

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

df.set_index(['id_number'], inplace= True)
print(df.head())

print(df.shape)

print(df.columns)
print(df.info())
print(df.isnull)
df['diagnosis']= df['diagnosis'].map({'M': 1, 'B': 0})

print(df['diagnosis'].value_counts())

def b_n_imbalance(_):
    i= 0
    n= 0
    Mal= 0
    Ben= 0
    for _ in df[_]:
        if(_ == 1):
            i+=1
        elif (_ ==0):
            n+=1
    Mal= (i/len(df)*100)
    Ben= (n/len(df)*100)
    print('The percentage of Malignant is {0:.2f}%'.format(Mal))
    print('The percentage of Benign is {0:.2f}%'.format(Ben))

print(df.describe())

b_n_imbalance('diagnosis')
sns.set()
corr= df.corr()
plt.figure(figsize= (10, 10))
sns.heatmap(corr, cbar= True, linewidths= .5, square= True, annot= False, annot_kws= {'size':15}, fmt='.2f', xticklabels= True, yticklabels= True, cmap= 'YlOrBr')
plt.show()
for i in col_names[1:]:
    if(df[i].std()>10):
        print('Deviation for ', i, 'is ', df[i].std())
train, test= train_test_split(df, test_size= 0.3, random_state= 101)
features= train.drop(['diagnosis'], 1)
labels= train['diagnosis']

cross_vals= []
for i in range(1, 50):
    if(i%2 != 0):
        knn= KNeighborsClassifier(n_neighbors= i, weights= 'distance', algorithm= 'kd_tree')
        scores= cross_val_score(knn, features, labels, cv=10, scoring= 'accuracy')
        cross_vals.append(scores.mean())
        
cross_vals_e= list(1 - np.array(cross_vals))
k_list= [i for i in range(1, 50) if(i%2)!=0]
optimal_k= k_list[cross_vals_e.index(min(cross_vals_e))]
print(optimal_k)

for w in ['uniform', 'distance']:
    knn= KNeighborsClassifier(n_neighbors= optimal_k, weights= w)
    fit= knn.fit(features, labels)
    pred= knn.predict(features)
    c_t= pd.crosstab(pred, labels, rownames= ['Predicted Values'], colnames= ['Actual Values'])
    print(c_t)
    acc_knn= knn.score(features, labels)
    print(acc_knn)
