from gensim import corpora, models
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')
k = 50
u_or_i = 'user'
path = '/data/lxkui/Amazon/office/'
# path = '../lda/'
"""
user_embedding_vec.csv:    user     reviews_emb
item_embedding_vec.csv:    item     reviews_emb
three.csv:               user-item  review_emb
"""

def get_theta_list(desc):
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for
                       document in desc]
    english_punctuations = [',', '.', ':', ';', '?', '!', '(', ')', '[', ']', '@', '&', '#', '%', '$', '{', '}', '--',
                            '-']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_tokenized]

    # Returning stems, e.g. created and creating both become create
    st = PorterStemmer()
    texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]

    # drop stopwords
    english_stopwords = stopwords.words('english')
    texts_cleaned = [[word for word in document if not word in english_stopwords] for document in texts_stemmed]
    print('111111111')
    dictionary = corpora.Dictionary(texts_cleaned)
    corpus = [dictionary.doc2bow(text) for text in texts_cleaned]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print('222222222')
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=k, alpha=50 / k)
    theta = lda.get_document_topics(corpus, minimum_probability=0.000001)
    theta_list = [[a[1] for a in b] for b in theta]
    return theta_list


# output pathUserReviews.out

list_read_path = ['user_text.csv','item_text.csv','two.csv']
list_save_path = ['user_embedding_vec.csv','item_embedding_vec.csv','three.csv']

# for i in range(3):
i = 1
df = pd.read_csv(path + 'json2csv/' + list_read_path[i], header=None, delimiter=',', index_col=None)
df = df.dropna(axis=0, how='any')
if i != 2: # user or item
    desc = df[1]
    df.drop(1, axis=1, inplace=True)
else: # two -> three
    desc = df[3]
    df.drop(3, axis=1, inplace=True)
theta_list = get_theta_list(desc)
print('333333333')
text_embedding = pd.DataFrame(theta_list)
df = pd.concat([df, text_embedding], axis=1, ignore_index=True)
df.to_csv(path + 'embedding/'+list_save_path[i], header=False, index=False, float_format='%.8f')



