# Twitter-Sentiment-Analysis

Swapnil Athawale (B15CS038)

# Motivation:
To get the opinion of peoples on any topic/product/issue.
Any company would want to know how people are reacting to its product.
Any political party would want to know whether people are supporting or not.
 # Objective:
   To determine whether the expressed tweet is expressing positive or
negative view about that topic. To classify the given tweet into positive or negative
category.
# Challenges/Research Issues:
To find the best dataset for training and testing.

To try various feature extraction techniques.
1) bag of words model
2) word2vec model
3) CountVectorizer

To try 2 different approaches for predicting sentiment.
1) Lexicon based approach
2) Machine Learning

To try out various models.
1) Linear Regression
2) Naive Bayes
3) Sequential Model

# Methodology/Algorithm:
Preprocess tweet into tokens. Remove hashtag, hyperlinks, stopwords.
Used word2vec neural network model to convert every token into vector.
Multiplying every word with its tf-idf score and taking average over the sum of all
tokens(words) in specific tweet to make sent2vec (vectors for sentence).
Training on sequential deep neural network (using Keras).
Collecting new tweets from Twitter to predict sentiment.
Display the predictions on webpage.

# Results
We were able to achieve 2 models
1) 85% accuracy on
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
2) 76% accuracy on http://www.sananalytics.com/lab/twitter-sentiment/

# Conclusion:
1) Dataset we provide has to be larger than mentioned above.
2) It performs fairly well within same dataset.
3) Though it performs very well within the same dataset it doesn’t perform very
well with new random tweets.
4) Deep Neural Network perform better than other machine learning models like
Naive Bayes.
# References:
   # Datasets:
1) Dataset 1 from stanford contains 1.6 million tweets with sentiment derived by
emoticon.
2) Dataset 2 from Sanders Analysis contains 1.57 million tweets with
hand-classified sentiments.
   # Articles:
1) https://www.youtube.com/watch?v=si8zZHkufRY&t=501s
2) https://blog.griddynamics.com/creating-training-and-test-data-sets-and-preparing-the-data-for-twitter-stream-sentiment-analysis-of-social-movie-reviews/
3) http://www.awesomestats.in/python-sentiment-tweets/
4) http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
5) https://realpython.com/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/
6) https://machinelearningmastery.com/clean-text-machine-learning-python/
7) https://code.tutsplus.com/tutorials/creating-a-web-app-from-scratch-using-python-flask-and-mysql--cms-22972
   # Research papers:
1) A survey of sentiment analysis techniques
https://ieeexplore.ieee.org/document/8058315/
2) A Comparison between Preprocessing Techniques for Sentiment Analysis in
Twitter ​ https://ceur-ws.org/Vol-1748/paper-06.pdf
