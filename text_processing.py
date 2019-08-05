import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all = pd.concat([train,test],axis=0)
comment_text = all.comment_text
target = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]

### 1. Comments Statistics
def preprcessing_cleaning(comment):
    comments_text_clean = []

    for sec in comment:
        temp = sec.split(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        temp2 = []

        # sec = st.stem(sec)

        for c in sec:
            if (c in ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~']) | (c in stop_words):
                c = c.replace(c, '')

            if len(c)>0:
                temp2.append(c)

        comments_text_clean.append([" ".join(x for x in temp2)])
    return comments_text_clean

def preprcessing_statistic(comment):
    temp = comment.split()
    pre_res = []
    ### N of words
    words = len(temp)
    Capital = punc = words = numbers = digits = alpha = stop = 0
    for c in temp:
        c = c.replace("\n",'')
        ### N of Capital
        Capital += c.isupper()
        ### N of punctions
        punc += (c in ['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'])
        ## N of characters
        words += len(temp)
        ## N of numbers
        numbers += c.isnumeric()
        ## N of digits
        digits += c.isdigit()
        ## N of Alpha
        alpha += c.isalpha()
        ## N of stopwords
        stop += (c in stop_words)

    alpha = words - alpha

    pre_res.append([words,Capital,punc,words,numbers,digits,alpha,stop])

    return pre_res


# comments_text_clean = preprcessing_cleaning(comment_text)

preprcessing_statistic_res = comment_text.map(lambda x: preprcessing_statistic(x))
preprcessing_statistic_res = np.asarray(tuple(tuple(x[0]) for x in preprcessing_statistic_res))

train_preprcessing_statistic_res = preprcessing_statistic_res[:train.shape[0],:]
test_preprcessing_statistic_res = preprcessing_statistic_res[train.shape[0]:,:]

pd.DataFrame(train_preprcessing_statistic_res).to_pickle('train_preprcessing_statistic_res.pkl')
pd.DataFrame(test_preprcessing_statistic_res).to_pickle('test_preprcessing_statistic_res.pkl')

### 2. tf-idf
tfidf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1))
preprcessing_tfidf_res = tfidf.fit_transform(comment_text)

svd = TruncatedSVD(n_components=200)
preprcessing_tfidf_res = svd.fit_transform(preprcessing_tfidf_res)

train_preprcessing_tfidf_res = preprcessing_tfidf_res[:train.shape[0],:]
test_preprcessing_tfidf_res = preprcessing_tfidf_res[train.shape[0]:,:]

pd.DataFrame(train_preprcessing_tfidf_res).to_pickle('train_preprcessing_tfidf_res.pkl')
pd.DataFrame(test_preprcessing_tfidf_res).to_pickle('test_preprcessing_tfidf_res.pkl')

### 3. topic modeling
countV = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,3), max_features=10000)
preprcessing_count_res = countV.fit_transform(comment_text)

lda = LatentDirichletAllocation(n_topics=10, max_iter=5, learning_offset=50.,random_state=0).fit(preprcessing_count_res)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, countV.get_feature_names(), no_top_words)
preprcessing_topic_res = lda.transform(preprcessing_count_res)

train_preprcessing_topic_res = preprcessing_topic_res[:train.shape[0],:]
test_preprcessing_topic_res = preprcessing_topic_res[train.shape[0]:,:]

pd.DataFrame(train_preprcessing_topic_res).to_pickle('train_preprcessing_topic_res.pkl')
pd.DataFrame(test_preprcessing_topic_res).to_pickle('test_preprcessing_topic_res.pkl')

###
train_preprocessing = np.concatenate([train_preprcessing_statistic_res,train_preprcessing_tfidf_res,train_preprcessing_topic_res],axis=1)
test_preprocessing  = np.concatenate([test_preprcessing_statistic_res,test_preprcessing_tfidf_res,test_preprcessing_topic_res],axis=1)

pd.DataFrame(train_preprocessing).to_pickle('train_preprocessing.pkl')
pd.DataFrame(test_preprocessing).to_pickle('test_preprocessing.pkl')
