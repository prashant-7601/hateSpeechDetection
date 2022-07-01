# import modules
from flask import Flask, request, render_template
import json, os
import pandas as panda
import regex as re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier)
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from werkzeug.utils import secure_filename


# global declarations and initializations
app = Flask(__name__)
stopwords = stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
lemm = WordNetLemmatizer()
hate_bag = {}
abuse_bag = {}


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
himages = {
    0: "https://st.depositphotos.com/1775533/1288/i/600/depositphotos_12880120-stock-photo-green-check-mark.jpg",
    1: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSukZWmlShXThaaA9kMDE2lUN5qJA5n4wxdDQ&usqp=CAU",
    2: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSukZWmlShXThaaA9kMDE2lUN5qJA5n4wxdDQ&usqp=CAU"
}
aimage = {
    0: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSukZWmlShXThaaA9kMDE2lUN5qJA5n4wxdDQ&usqp=CAU",
    1: "https://st.depositphotos.com/1775533/1288/i/600/depositphotos_12880120-stock-photo-green-check-mark.jpg",
    2: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSukZWmlShXThaaA9kMDE2lUN5qJA5n4wxdDQ&usqp=CAU"
}
# function for preprocessing all tweets in the dataset
def preprocess(tweet, sentiment):  
    regex_pat = "\s+"
    tweet_space = tweet.str.replace(regex_pat, ' ')
    regex_pat = "@[\w\-]+'"
    tweet_name = tweet_space.str.replace(regex_pat, '')
    giant_url_regex =  "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    tweets = tweet_name.str.replace(giant_url_regex, '')
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    newtweet=punc_remove.str.replace(r'\s+', ' ')
    newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
    newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
    tweet_lower = newtweet.str.lower()
    tokenized_tweet = tweet_lower.apply(lambda x: x.split())
    tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
    tokenized_tweet = tokenized_tweet.apply(lambda x: [lemm.lemmatize(i) for i in x]) 
    
    for i in range(len(tokenized_tweet)):
        if sentiment[i] == 0:
            for word in tokenized_tweet[i]:
                if word not in hate_bag:
                    hate_bag[word] = 1
                else:
                    hate_bag[word] += 1
        elif sentiment[i] == 1:
            for word in tokenized_tweet[i]:
                if word not in abuse_bag:
                    abuse_bag[word] = 1
                else:
                    abuse_bag[word] += 1
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p= tokenized_tweet
    
    return tweets_p

# function to preprocess input string
def preprocess1(tweet):  
    regex_pat = re.compile(r'\s+')
    tweet_space = re.sub(regex_pat, ' ', tweet)
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = re.sub(regex_pat, '', tweet_space)
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = re.sub(giant_url_regex, '', tweet_name)
    punc_remove = re.sub("[^a-zA-Z]", " ", tweets)
    newtweet= re.sub(r'\s+', ' ', punc_remove)
    newtweet= re.sub(r'^\s+|\s+?$','',newtweet)
    newtweet= re.sub(r'\d+(\.\d+)?','numbr', newtweet)
    tweet_lower = newtweet.lower()
    tokenized_tweet = tweet_lower.split()
    tokenized_tweet=  [item for item in tokenized_tweet if item not in stopwords]
    tokenized_tweet = [lemm.lemmatize(i) for i in tokenized_tweet]
    
    tokenized_tweet = ' '.join(tokenized_tweet)
    
    return tokenized_tweet


# input and preprocess data
dataset = panda.read_csv("data2.csv")
processed_tweets = preprocess(dataset.tweet, dataset['class'])
dataset['processed_tweets'] = processed_tweets
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(dataset['processed_tweets'] )

# split training and testing data
X = tfidf
y = dataset['class'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# declare models for ensemble learning
models = [ 
    ('Naive Bayes', BernoulliNB()),
    ('Decision Tree', tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy")),
    ('Logistic Regression', LogisticRegression(solver='lbfgs', max_iter=1000)),
    ('Random Forest', RandomForestClassifier(n_jobs = -1, n_estimators = 500, warm_start = True, min_samples_leaf = 2, max_features = 'sqrt', verbose = 0))
]

# build and train model
final_model = StackingClassifier([models[1], models[2], models[3]], final_estimator=LogisticRegression(solver='lbfgs', max_iter=1000))
final_model.fit(x_train, y_train)

# API routes
@app.route("/filepredict", methods=["POST"])
def filepredict():
    if 'myfile' not in request.files:
        return "no file selected"
    file = request.files['myfile']
    if file.filename == '':
        return "no file selected"
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = (open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "r")).read()
        text = preprocess1(text)
        words = text.split()
        text = tfidf_vectorizer.transform([text])
        result = final_model.predict(text)
        probabilities = final_model.predict_proba(text)
        d = {0: "Hate", 1: "Abusive", 2: "Neutral"}
        hate_words = []
        abuse_words = []
        for word in words:
            if word in abuse_bag and word not in abuse_words:
                abuse_words.append(word)
            if word in hate_bag and word not in hate_words:
                hate_words.append(word) 
        return render_template("result.html", hateimage=himages[result[0]], abuseimage=aimage[result[0]], hate_prob = f'{probabilities[0][0]*100:.2f}%', abuse_prob = f'{probabilities[0][1]*100:.2f}%', hate_words = hate_words, abuse_words = abuse_words)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["name"]
    text = preprocess1(text)
    words = text.split()
    text = tfidf_vectorizer.transform([text])
    result = final_model.predict(text)
    probabilities = final_model.predict_proba(text)
    d = {0: "Hate", 1: "Abusive", 2: "Neutral"}
    hate_words = []
    abuse_words = []
    for word in words:
        if word in abuse_bag and word not in abuse_words:
            abuse_words.append(word)
        if word in hate_bag and word not in hate_words:
            hate_words.append(word)  
    if result[0] !=0:
        hate_words = []
    if result[0] != 1:
        abuse_words = []      
    return render_template("result.html", hateimage=himages[result[0]], abuseimage=aimage[result[0]], hate_prob = f'{probabilities[0][0]*100:.2f}%', abuse_prob = f'{probabilities[0][1]*100:.2f}%', hate_words = hate_words, abuse_words = abuse_words)

@app.route("/")
def hello():
  return render_template("home.html")


if __name__ == "__main__":
  app.run()