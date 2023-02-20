import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans



data=pd.read_csv("Creditcard_data.csv")

data.shape


data.Class.value_counts()


X = data.drop('Class',axis=1)
Y = data['Class']

print(X.shape)
print(Y.shape)

from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(random_state=0)
X_train_os, Y_train_os = rus.fit_resample(X,Y)

print("Before Over Sampling :" , Counter(Y))
print("After Over Sampling :" , Counter(Y_train_os))

balanced_dataframe = pd.concat([X_train_os, Y_train_os], axis=1)
balanced_dataframe.to_csv('balanced_dataset.csv', index=False)
print("Balanced Dataset Created...")


p = np.sum(Y_train_os) / len(Y_train_os)


confidence_level = input("Enter the Confidence Level(in %) :")  
confidence_level = float(confidence_level)/100

alpha = 1-confidence_level
print("The Margin of Error :",alpha)

z_score = norm.ppf(1-alpha/2)
print("Z-Score is :",z_score)

n = int(np.ceil((z_score*2 * p * (1-p)) / (alpha*2)))
print("Sample Size is :",n)

sampling_data = []

sample0 = balanced_dataframe.sample(n, replace=False)
sampling_data.append(sample0)

sample1=balanced_dataframe.groupby('Class',group_keys=False).apply(lambda x: x.sample(frac=.2523))
sampling_data.append(sample1)


sampling_interval = int(len(balanced_dataframe) / n) 

indices = np.arange(start=0, stop=len(balanced_dataframe), step=sampling_interval)[:n]

sample2 = balanced_dataframe.iloc[indices]
sampling_data.append(sample2)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(balanced_dataframe.iloc[:, :-1])
clusters = kmeans.predict(balanced_dataframe.iloc[:, :-1])
balanced_dataframe['cluster'] = clusters

proportions = balanced_dataframe['cluster'].value_counts(normalize=True)

desired_sample_size = n
sample_sizes = np.round(proportions * n).astype(int)

sample3 = []

for cluster, size in sample_sizes.iteritems():
    cluster_data = balanced_dataframe[balanced_dataframe['cluster'] == cluster]
    sample = cluster_data.sample(n=size, random_state=0)
    sample3.append(sample)

sample3 = pd.concat(sample3)

sample3 = sample3.drop('cluster', axis=1)
sampling_data.append(sample3)

zeros_df = balanced_dataframe[balanced_dataframe['Class'] == 0].sample(int(n/2), random_state=1)
ones_df = balanced_dataframe[balanced_dataframe['Class'] == 1].sample(int(n/2), random_state=1)

sample4 = pd.concat([zeros_df, ones_df])
sampling_data.append(sample4)

for i, sample in enumerate(sampling_data):
    sample.to_csv(f'sample_dataset_{i}.csv', index=False)

models = [
    Pipeline([('scaler', StandardScaler()),('lr', LogisticRegression(max_iter=1000))]),
    GaussianNB(),LDA(), SVC(), KNeighborsClassifier()]

model_names = ['Logistic Regression', 'Naive Bayes', 'LDA', 'SVC', 'KNN']

result_102003106 = pd.DataFrame(columns=['Dataset', *model_names])

for i, sample in enumerate(sampling_data):
    X = sample.iloc[:, :-1]
    y = sample.iloc[:, -1]
    row = {'Dataset': f'Sampling {i+1}'}
    for j, model in enumerate(models):
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        row[model_names[j]] = f'{accuracy:.3f}'
    result_102003106 = result_102003106.append(row, ignore_index=True)

result_102003106 = result_102003106.set_index('Dataset').T.rename_axis('Model', axis=0)

print(result_102003106)
result_102003106.to_csv(f'Final_Solution.csv', index=False)