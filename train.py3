import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np 
from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec 
LabeledSentence = gensim.models.doc2vec.LabeledSentence 
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
import re
from keras.models import Sequential 
from keras.layers import Dense
from copy import deepcopy
from string import punctuation
from random import shuffle

n=1000000
n_dim =200


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
    df = df.loc[df.ModiSentimentText.str.len() != 0,:]
    #df.Sentimentext = texts 
    data['tokens'] = texts  


    return df


#read csv file here  and name it as data 


x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens), np.array(data.head(n).Sentiment), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST') 

tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in x_train], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

#tweet_w2v.save('w2v_1.model')

print(tweet_w2v.most_similar('good'))

print ('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train,validation_split=0.30, epochs=9, batch_size=32, verbose=2)


score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print (score[1])
