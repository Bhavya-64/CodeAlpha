#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  # Import pandas first

df = pd.read_csv(r"E:\Internships\CodAlpha\archive\loan\loan.csv")
df.head()  # Show first 5 rows


# In[2]:


df.shape


# In[3]:


df.info()


# In[6]:


df.drop(df.columns.difference(['loan_amnt','term','int_rate','installment','grade','emp_length','home_ownership',
                                         'annual_inc','verification_status','loan_status','purpose',]),axis=1, inplace=True)


# In[7]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.annual_inc = df.annual_inc.fillna(0)
df.isnull().sum()


# In[11]:


label_categories = [
    (0, ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid', 'Current']),
    (1, ['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 
         'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off'])
]

# function to apply the transformation
def classify_label(text):
    for category, matches in label_categories:
        if any(match in text for match in matches):
            return category
    return None

df.loc[:, 'label'] = df['loan_status'].apply(classify_label)
df = df.drop('loan_status', axis=1)


# In[13]:


def SC_LabelEncoder1(text):
    if text == "E":
        return 1
    elif text == "D":
        return 2
    elif text == "C":
        return 3
    elif text == "B":
        return 4
    elif text == "A":
        return 5
    else:
        return 0


# In[14]:


def SC_LabelEncoder2(text):
    if text == "< 1 year":
        return 1
    elif text == "1 year":
        return 2
    elif text == "2 years":
        return 3
    elif text == "3 years":
        return 4
    elif text == "4 years":
        return 5
    elif text == "5 years":
        return 6
    elif text == "6 years":
        return 7
    elif text == "7 years":
        return 8
    elif text == "8 years":
        return 9
    elif text == "9 years":
        return 10
    elif text == "10 years":
        return 11
    elif text == "10+ years":
        return 12
    else:
        return 0


# In[15]:


def SC_LabelEncoder3(text):
    if text == "RENT":
        return 1
    elif text == "MORTGAGE":
        return 2
    elif text == "OWN":
        return 3
    else:
        return 0


# In[16]:


df["grade"] = df["grade"].apply(SC_LabelEncoder1)
df["emp_length"] = df["emp_length"].apply(SC_LabelEncoder2)
df["home_ownership"] = df["home_ownership"].apply(SC_LabelEncoder3)


# In[17]:


df.head(10)


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style('darkgrid')
sns.countplot(data=df, x='grade', hue='home_ownership', ax=ax[0], palette='Set2').set_title("Grade/Home Ownership distribution")
sns.countplot(data=df, x='term', hue='home_ownership', ax=ax[1], palette='Set2').set_title("Term/Home Ownership distribution")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style('darkgrid')
sns.countplot(data=df, x='grade', hue='verification_status', ax=ax[0], palette='Set2').set_title("Grade/Verification Status distribution")
sns.countplot(data=df, x='term', hue='verification_status', ax=ax[1], palette='Set2').set_title("Term/Verification Status distribution")


# In[22]:


fig, ax = plt.subplots(1,4,figsize=(20,5))
sns.histplot(df, x='loan_amnt',hue="label", bins=30, ax=ax[0],palette='Set2').set_title("Loan Ammount distribution");
sns.countplot(data=df, x='term', hue="label", ax=ax[1],palette='Set2').set_title("Term distribution");
sns.countplot(data=df, hue='home_ownership', x='label', ax=ax[2],palette='Set2').set_title("Home ownership with loan_status");
sns.countplot(data=df, x='verification_status', hue='label', ax=ax[3],palette='Set2').set_title("Verification Status Distribution with loan_status");


# In[23]:


sns.set(rc={'figure.figsize':(10,5)})
sns.heatmap(df[['loan_amnt', 'int_rate', 'grade', 'emp_length', 'home_ownership', 'annual_inc','label']].corr(),cbar=True,annot=True,
            linecolor='white',linewidths=1.5,cmap="mako").set_title("Pearson Correlations Heatmap");


# In[25]:


from sklearn.preprocessing import LabelEncoder
for col in ["verification_status", "purpose","term"]:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])


# In[28]:


X, y = df.drop("label", axis=1), df["label"]


# In[31]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


# In[36]:


acc = []
pre = []
f1 =[]
rec =[]


# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[39]:


knn.fit(X_train_scaled,y_train)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')  # faster search method
knn.fit(X_train_scaled, y_train)


# In[44]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
pred_logreg = logreg.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, pred_logreg))
print("Accuracy =", accuracy_score(y_test, pred_logreg))


# In[45]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


rftree = RandomForestClassifier(n_estimators=10)


# In[47]:


rftree.fit(X_train_scaled,y_train)


# In[48]:


rftree_pred = rftree.predict(X_test_scaled)


# In[49]:


print("Classification Report :")
print(classification_report(y_test,rftree_pred))


# In[50]:


print("Accuracy = ",accuracy_score(y_test,rftree_pred))


# In[51]:


acc.append(accuracy_score(y_test,rftree_pred))
pre.append(precision_score(y_test,rftree_pred))
rec.append(recall_score(y_test,rftree_pred))
f1.append(f1_score(y_test,rftree_pred))


# In[52]:


from sklearn.tree import DecisionTreeClassifier


# In[53]:


dtree = DecisionTreeClassifier()


# In[54]:


dtree.fit(X_train_scaled,y_train)


# In[55]:


pred_dtree = dtree.predict(X_test)


# In[56]:


print("Classification Report :")
print(classification_report(y_test,pred_dtree))


# In[57]:


print("Accuracy = ",accuracy_score(y_test,pred_dtree))


# In[58]:


acc.append(accuracy_score(y_test,pred_dtree))
pre.append(precision_score(y_test,pred_dtree))
rec.append(recall_score(y_test,pred_dtree))
f1.append(f1_score(y_test,pred_dtree))


# In[59]:


labels = ['KNN','Random Forest','Decision Tree']


# In[62]:


# Predict using KNN
pred_knn = knn.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, pred_knn))
print("Accuracy =", accuracy_score(y_test, pred_knn))

# Append to metrics
acc.append(accuracy_score(y_test, pred_knn))
pre.append(precision_score(y_test, pred_knn))
rec.append(recall_score(y_test, pred_knn))
f1.append(f1_score(y_test, pred_knn))


# In[ ]:





# In[ ]:




