import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()

# Let's plot out just the first 5 variables (features)
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )
df_cancer['target'].value_counts()
sns.countplot(df_cancer['target'], label = "Count")
plt.figure(figsize=(20,12)) 
sns.heatmap(df_cancer.corr(), annot=True)
