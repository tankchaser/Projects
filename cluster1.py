#!/usr/bin/env python
# coding: utf-8

# In[408]:


import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt


# In[431]:


df = pd.read_excel('/Users/tankchaser/Desktop/pr_1_yearequalmonth.xlsx')
df.head(50)


# In[432]:


df.info()


# In[420]:


dec = df.describe()
dfr = pd.DataFrame(dec) 
ay = pd.ExcelWriter('./description.xls') 
dfr.to_excel(ay) 
ay.save()
print(dec)


# # Проверка данных на пустые значения

# In[204]:


df.isnull().sum()


# In[ ]:


Company Age Number of Investors Cheapest Monthly Package Volume-based Price Fixed Price


# Category Region Company Age	Number of Employees	6 Month Growth	Number of Investors	Supported Languages	Price Availability Cheapest Monthly Package Most Expensive Monthly Package Cheapest Yearly Package Most Expensive Yearly Package Price Range Discount for Smallest Package Discount for Biggest Package Monthly Subscription Yearly Subscription Localization Customization	Freemium Free Trial	Number of Versions Segmentation	Per Feature	Per User One Time	Pay As You Go	Volume-based Price	Fixed Price

# In[ ]:


'Category','Region','Number of Employees','6 Month Growth','Supported Languages','Price Availability','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go'


# In[ ]:





# In[447]:


df.hist('Number of Employees', bins=10, figsize=(15,10))
plt.xlabel('Категория', fontsize=30)
plt.ylabel('Количество', fontsize=30)
plt.title('Регион', fontsize=30)


# In[280]:


import seaborn as sns


# In[206]:


df.hist(bins=50, figsize=(30,25))#графический анализ по частотному графику


# In[128]:


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df)


# In[571]:


import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

ax.scatter(df, df)    #  цвет точек

plt.show()


# In[448]:


correlation_matrix = df[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']].corr().round(2)

fig, ax = plt.subplots(figsize=(25,24)) 
sns.heatmap(data=correlation_matrix, annot=True,cmap='RdYlGn')


# correlation_matrix = df.corr(method='spearman').round(2)
# 
# fig, ax = plt.subplots(figsize=(25,24)) 
# sns.heatmap(data=correlation_matrix, annot=True)

# In[449]:


correlation_matrix = df[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']].corr(method='spearman').round(2)

fig, ax = plt.subplots(figsize=(25,24)) 
sns.heatmap(data=correlation_matrix, annot=True,cmap='RdYlGn')


# In[352]:


from sklearn import preprocessing
import pandas as pd

df_norm = df[['Category', 'Region','Company Age','Number of Employees', '6 Month Growth', 'Number of Investors', 'Supported Languages','Price Availability','Cheapest Monthly Package','Most Expensive Monthly Package','Cheapest Yearly Package','Price Range','Most Expensive Yearly Package','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization', 'Freemium', 'Free Trial', 'Number of Versions','Segmentation', 'Per Feature', 'Per User', 'One Time', 'Pay As You Go', 'Volume-based Price', 'Fixed Price']]
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

names = df_norm.columns
d = scaler.fit_transform(df_norm)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()


# In[348]:


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, AffinityPropagation
from sklearn.metrics import adjusted_rand_score, silhouette_score


# In[ ]:





# In[ ]:





# In[349]:


df_marks= df[['Name']]
rez = pd.concat([df_marks,scaled_df],axis=1)
rez.head()


# In[527]:


pip install pyclustertend


# In[535]:


from sklearn import datasets
from pyclustertend import hopkins

hopkins(scaled_df,300)


# In[537]:


from pyclustertend import vat
from pyclustertend import ivat
from sklearn.preprocessing import scale
X = scale(part_50[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']])
vat(X)


# In[350]:


# create kmeans object
kmeans = KMeans(n_clusters=2)
# fit kmeans object to data
kmeans.fit(scaled_df)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
X = kmeans.cluster_centers_
y_km = kmeans.fit_predict(scaled_df)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


# In[ ]:


##Calinski-Harabasz Index


# In[247]:


from sklearn.metrics.cluster import calinski_harabasz_score
calinski_harabasz_score(scaled_df, labels)


# In[353]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(scaled_df)
print(kmeans.cluster_centers_)
X = kmeans.cluster_centers_
y_km = kmeans.fit_predict(scaled_df)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


# In[249]:


from sklearn.metrics.cluster import calinski_harabasz_score
calinski_harabasz_score(scaled_df, labels)


# In[ ]:





# In[250]:


rez.head()

['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Cheapest Monthly Package','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']     
                
# In[510]:


import scipy.cluster.hierarchy as shc
# Plot
plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Дендограмма", fontsize=30)  
dend = shc.dendrogram(shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward'),orientation='top', labels=rez.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[602]:


import scipy.cluster.hierarchy as shc
# Plot
plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Дендограмма", fontsize=30)  
dend = shc.dendrogram(shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], metric='euclidean', method='complete'),orientation='top', labels=rez.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[503]:


import scipy.cluster.hierarchy as shc
# Plot
plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Дендограмма", fontsize=30)  
dend = shc.dendrogram(shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='median'),orientation='top', labels=rez.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[511]:


from scipy.cluster.hierarchy import fcluster
link = shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward')
rez['cluster'] = fcluster(link,10.2,criterion = 'distance')
rez.groupby('cluster').mean()


# In[590]:


import gower
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

dm = gower.gower_matrix(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']])
Zd = linkage(dm)


# In[591]:



plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Дендограмма", fontsize=30)  
dendrogram(Zd)
plt.xticks(fontsize=10)
plt.show()


# In[483]:


pip install gower


# In[339]:


import scipy.cluster.hierarchy as shc
# Plot
plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Dendograms", fontsize=22)  
dend = shc.dendrogram(shc.linkage(rez[['Category','Region','Number of Employees','6 Month Growth','Supported Languages','Price Availability','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go']], method='ward'),orientation='right', labels=rez.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[340]:


import scipy.cluster.hierarchy as shc
# Plot
plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Dendograms", fontsize=22)  
dend = shc.dendrogram(shc.linkage(rez[['Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Monthly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward'),orientation='right', labels=rez.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[399]:


# Creating a dataframe with 50%
# values of original dataframe
part_50 = rez.sample(frac = 0.5)
  
# Creating dataframe with 
# rest of the 50% values
rest_part_50 = rez.drop(part_50.index)


# In[403]:


plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Dendograms", fontsize=22)  
dend = shc.dendrogram(shc.linkage(part_50[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Cheapest Monthly Package','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward'),orientation='right', labels=part_50.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[404]:


plt.figure(figsize=(40, 50), dpi= 80)  
plt.title("Dendograms", fontsize=22)  
dend = shc.dendrogram(shc.linkage(rest_part_50[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Cheapest Monthly Package','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward'),orientation='right', labels=rest_part_50.Name.values)  
plt.xticks(fontsize=10)
plt.show()


# In[ ]:





# In[260]:


from scipy.cluster.hierarchy import fcluster
link = shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Cheapest Monthly Package','Most Expensive Monthly Package','Cheapest Yearly Package','Most Expensive Yearly Package','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward')
rez['cluster'] = fcluster(link,10,criterion = 'distance')
rez.groupby('cluster').mean()


# In[605]:


from scipy.cluster.hierarchy import fcluster
link = shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], metric='euclidean', method='complete')
rez['cluster'] = fcluster(link,3.3,criterion = 'distance')
rez.groupby('cluster').mean()


# In[594]:


from scipy.cluster.hierarchy import fcluster
rez['cluster'] = fcluster(Zd,2,criterion = 'distance')
rez.groupby('cluster').mean()


# In[566]:


rez.groupby('cluster').size()


# In[254]:


link


# In[607]:



dataframe = pd.DataFrame(rez_mean) 
ew = pd.ExcelWriter('./test8.xls') 
dataframe.to_excel(ew) 
ew.save() 


# In[608]:


df_mean = pd.read_excel('/Users/tankchaser/Desktop/test8.xls')
df_m = df_mean[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization', 'Freemium', 'Free Trial', 'Number of Versions','Segmentation', 'Per Feature', 'Per User', 'One Time', 'Pay As You Go', 'Volume-based Price', 'Fixed Price']]
fig, ax = plt.subplots(figsize=(25,5))
plt.title("Средние значения переменных в кластерах", fontsize=22)  
sns.heatmap(df_m, vmin=0, vmax=1,cmap='RdYlGn')


# In[606]:


rez_mean=rez.groupby('cluster').mean()
print(rez_mean)
sns.barplot(x=rez['cluster'], y=link, data=rez_mean)


# In[ ]:





# In[ ]:





# In[518]:


link = shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward')
rez['cluster'] = fcluster(link,10.2,criterion = 'distance')
df_rez = rez['cluster'] 
df_rez_new = pd.concat([df_rez,rez],axis=1)
df_rez_new.head()
dataframe1 = pd.DataFrame(df_rez_new) 
cl = pd.ExcelWriter('./test2.xls') 
dataframe1.to_excel(cl) 
cl.save() 


# In[ ]:





# In[ ]:


#ANOVA


# In[310]:


from scipy import stats
clus_1 = rez[rez['cluster']==1]['Region']

clus_2 = rez[rez['cluster']==2]['Region']

clus_3 = rez[rez['cluster']==3]['Region']

clus_4 = rez[rez['cluster']==4]['Region']

clus_5 = rez[rez['cluster']==5]['Region']

stats.f_oneway(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[311]:


clus_1 = rez[rez['cluster']==1]['Free Trial']

clus_2 = rez[rez['cluster']==2]['Free Trial']

clus_3 = rez[rez['cluster']==3]['Free Trial']

clus_4 = rez[rez['cluster']==4]['Free Trial']

clus_5 = rez[rez['cluster']==5]['Free Trial']

stats.f_oneway(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[312]:


clus_1 = rez[rez['cluster']==1]['Number of Versions']

clus_2 = rez[rez['cluster']==2]['Number of Versions']

clus_3 = rez[rez['cluster']==3]['Number of Versions']

clus_4 = rez[rez['cluster']==4]['Number of Versions']

clus_5 = rez[rez['cluster']==5]['Number of Versions']

stats.f_oneway(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[314]:


clus_1 = rez[rez['cluster']==1]['Per Feature']

clus_2 = rez[rez['cluster']==2]['Per Feature']

clus_3 = rez[rez['cluster']==3]['Per Feature']

clus_4 = rez[rez['cluster']==4]['Per Feature']

clus_5 = rez[rez['cluster']==5]['Per Feature']

stats.f_oneway(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[315]:


clus_1 = rez[rez['cluster']==1]['Fixed Price']

clus_2 = rez[rez['cluster']==2]['Fixed Price']

clus_3 = rez[rez['cluster']==3]['Fixed Price']

clus_4 = rez[rez['cluster']==4]['Fixed Price']

clus_5 = rez[rez['cluster']==5]['Fixed Price']

stats.f_oneway(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[316]:


clus_1 = rez[rez['cluster']==1]['Fixed Price']

clus_2 = rez[rez['cluster']==2]['Fixed Price']

clus_3 = rez[rez['cluster']==3]['Fixed Price']

clus_4 = rez[rez['cluster']==4]['Fixed Price']

clus_5 = rez[rez['cluster']==5]['Fixed Price']

stats.kruskal(clus_1, clus_2, clus_3, clus_4, clus_5)


# In[ ]:





# In[623]:


from sklearn.metrics import silhouette_score

scores = [0]
for i in range(2,11):
    fitx = KMeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(scaled_df)
    score = silhouette_score(scaled_df, fitx.labels_)
    scores.append(score)
    
plt.figure(figsize=(11,8.5))
plt.plot(range(1,11), np.array(scores), 'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Average Silhouette')
plt.title('The Elbow Method showing the optimal $k$')
plt.show()


# In[618]:


k_inertia = []
ks = range(1,15)

for k in ks:
    clf_kmeans = KMeans(n_clusters=k)
    clusters_kmeans = clf_kmeans.fit_predict(scaled_df, )
    k_inertia.append(clf_kmeans.inertia_)


# In[619]:


plt.plot(ks, k_inertia)
plt.plot(ks, k_inertia ,'ro')


# In[620]:


diff = np.diff(k_inertia)
plt.plot(ks[1:], diff)


# In[621]:


diff_r = diff[1:] / diff[:-1]
plt.plot(ks[1:-1], diff_r)


# In[378]:


k_opt = ks[np.argmin(diff_r)+1]
k_opt


# In[ ]:


# Compute distance matrix
dist= dist(scaled_df, method = "euclidean")

# Compute 2 hierarchical clusterings
hc1 <- hclust(res.dist, method = "complete")
hc2 <- hclust(res.dist, method = "ward.D2")

# Create two dendrograms
dend1 <- as.dendrogram (hc1)
dend2 <- as.dendrogram (hc2)

tanglegram(dend1, dend2)


# In[615]:


import tanglegram as tg

dend1 = shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='complete')  
dend2 = shc.linkage(rez[['Category','Region','Company Age','Number of Employees','6 Month Growth','Number of Investors','Supported Languages','Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward')  
fig = tg.gen_tangle(dend1, dend2, optimize_order=False)
plt.show()


# In[622]:


dend1 = shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='complete')  
dend2 = shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='ward')  
fig = tg.gen_tangle(dend1, dend2, optimize_order=False)
plt.show()


# In[625]:


dend1 = shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='complete')  
dend2 = shc.linkage(rez[['Price Availability','Price Range','Discount for Smallest Package','Discount for Biggest Package','Monthly Subscription','Yearly Subscription','Localization','Customization','Freemium','Free Trial','Number of Versions','Segmentation','Per Feature','Per User','One Time','Pay As You Go','Volume-based Price','Fixed Price']], method='weighted')  
fig = tg.gen_tangle(dend1, dend2, optimize_order=False)
plt.show()


# In[610]:


pip install tanglegram -U


# In[ ]:




