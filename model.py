#https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -->kaggle link of the dataset

import pandas as pd
import re

df=pd.read_csv('IMDB Dataset.csv')
df=df.sample(30000)
df['sentiment'].replace({'positive':1,'negative':0},inplace=True)

# Function to clean html tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
df['review']=df['review'].apply(clean_html)

# converting everything to lower
def convert_lower(text):
    return text.lower()
df['review']=df['review'].apply(convert_lower)

# function to remove special character
def remove_special(text):
    x=''
    
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x + ' '
    return x
df['review']=df['review'].apply(remove_special)

from nltk.corpus import stopwords

def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y

df['review']=df['review'].apply(remove_stopwords)

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
df['review']=df['review'].apply(stem_words)

def join_back(list_input):
    return " ".join(list_input)
df['review']=df['review'].apply(join_back)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)

X=cv.fit_transform(df['review']).toarray()
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

clf1=GaussianNB()
clf2=MultinomialNB()
clf3=BernoulliNB()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)

y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)

from sklearn.metrics import accuracy_score

print("Gaussian",accuracy_score(y_test,y_pred1))
print("Multinomial",accuracy_score(y_test,y_pred2))
print("Bernaulli",accuracy_score(y_test,y_pred3))

import pickle

pickle.dump(clf3,open('model.pkl','wb'))
print("Successfully dumped")


