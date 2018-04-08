import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
import numpy as np
import pickle
import tensorflow
import keras

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class

#import gensim.utils.SaveLoad
'''
>>> model.save(fname)# fname=w2vname.model
>>> model = Word2Vec.load(fname)
'''

LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stoplist = set(stopwords.words("english"))

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

def preprocessing(df):
    texts = df.SentimentText.tolist() 
    texts = [re.sub(r'http\S+|@\S+|#\S+' , '',text, flags=re.MULTILINE) for text in texts] #remove links and hashtags
    #texts = [ str(text,'ascii','ignore') for text in texts]  # ignoring emoji's unicode, do it only once repeating this will cause error 
    texts = [ str(text) for text in texts]  # ignoring emoji's unicode, do it only once repeating this will cause error 

    texts = [re.sub(r'^\d+\s|\s\d+\s|\s\d+$|\d\S' , '',text, flags=re.MULTILINE) for text in texts] #removing numbers 
    
    from nltk.tokenize import word_tokenize # puntk and stopwords data is in home/swapnil
    texts = [word_tokenize(text) for text in texts]
    #texts = [text  not in stoplist for text in texts]
    
    texts = [[ word.lower() for word in text if word.isalpha()] for text in texts ]
    texts = [[ word  for word  in text if word  not in stoplist] for text in texts ]
    '''
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    texts = [[ porter.stem(word) for word in text if word.isalpha()] for text in texts ]
    '''
    df['ModiSentimentText'] = texts
    #df = df.loc[df.ModiSentimentText.str.len() != 0,:] 
    #df.loc[:,'ModiSentimentText']=df.ModiSentimentText.apply(lambda x: ' '.join(x) )

    return df
#nrows= 1000000
df = pd.read_csv('finalizedfull.csv', header=None, error_bad_lines=False)
df.rename(columns= {0:'Author', 1:'SentimentText'}, inplace=True)


#df.reset_index(drop=True, inplace=True)

df = preprocessing(df)
df = df[df.ModiSentimentText.isnull() == False]
df.reset_index(drop=True, inplace=True)
#df.rename(columns= {0:'Sentiment', 1:'SentimentText', 2:'ModiSentimentText'}, inplace=True)

print('preprocessing done')


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
x_test = labelizeTweets(df.ModiSentimentText, 'TEST') 


n_dim=200
w2v = Word2Vec(size=n_dim, min_count=3)
w2v.build_vocab([x.words for x in x_test])
w2v.train([x.words for x in x_test], total_examples=w2v.corpus_count, epochs=w2v.iter)


vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=2)
matrix = vectorizer.fit_transform([x.words for x in x_test])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def buildWordVector(tok, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    #print(tok)
    for word in tok:
        #print('word  ', word)
        try:
            vec += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec




from sklearn.preprocessing import scale
test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)
from keras.models import load_model
model = load_model('76_model.h5')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

preds = model.predict_classes(test_vecs_w2v, batch_size=128, verbose=2)

assert(df.shape[0]==preds.shape[0])

df['Prediction'] = preds

from sklearn.metrics import accuracy_score
acc = accuracy_score(preds, df.Sentiment)
print(acc)