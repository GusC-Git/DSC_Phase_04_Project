# Classification of Tweet Sentiment using NLP

* Non-technical slide presentation: https://docs.google.com/presentation/d/1660HWukx0fL_0tCbnfBuSO2z8h9PYZ7UxDAVsMzAa9w/edit?usp=sharing
* Non-technical video presentation:
---------------------------
## Predicting product sentiment for companies based on Twitter data
----------------------------
![](Images/sxswlogo.png)
SXSW logo [Source](https://music.mxdwn.com/2018/02/15/news/house-of-vans-spotify-and-more-major-brands-will-not-be-holding-events-at-sxsw-this-year/)
## Goals and Overview
Twitter is a powerful social media platform that connects millions of users to each other daily. One of the best things of our modern societal age is the ability to easily and quickly share information with others through platforms such as Twitter. This can be used to companies' advantage in the form of instant feedback; by analyzing what it is that people are saying on twitter, a company is able to gather information about how people felt about a certain event, product, or the company in general and this can help inform business decisions. This project uses twitter data that was gathered during the time of the SXSW Event in 2001 about Apple and Google products and aims to build a classifier that can correctly classify the emotion within these tweets. The data is currently hosted by Crowdflower and can be found [here](https://data.world/crowdflower/brands-and-product-emotions)

# Preprocessing the Data
The Preprocessing step when working with text data is the step that offers the most challenges. In addition to normal preprocessing steps such as dealing with missing values, it is imperative to discern how specifically we will extract the text data that we will be working with.

## Columns and Descriptions
I renamed all columns so that they were easier to work with. Column names will be presented in the format of Original Name -> New Name - Description of column.

* tweet_text -> tweet -> string of characters found in tweet
* emotion_in_tweet_is_directed_at -> product - the product at which the tweet was directed at
* is_there_an_emotion_at_a_brand_or_product -> emotion - our target class. sentiment found in tweet

## Dealing with missing data
There did happen to be some missing data within this set. Specifically, our product column contained more than half of it's data as missing. 
![](Images/missingdata.png)
I decided to drill in and investigate to see if there was some information in the fact that the data was missing. 
![](Images/missingdataclassdist.png)
In the graph above we see what percentage of each class in the total dataset is represented within the subset where there is no product that which the emotion is targeted at. What I found is that where there is no specific product listed, there was a significantly higher chance of there being no emotion, and that the overwhelming majority of tweets that gave neither positive nor negative sentiment were about no product in particular. All tweets classified as "I can't tell" I simply dropped from the set. Additionally, I reduced all products into 3 different classes: Apple, Google, No specific brand.

## Tokenizing Data with Regular Expressions
For the next step I needed to reduce my text data to only contain the important words of each tweet. For this, I used one regular expression to remove mentions and another to tokenize the remaining words. Afterwards I removed all stopwords as well as 'sxsw' and lemmatized the remaining words. I did this all within a function I named preprocess.
```
def preprocess(X):
    """Takes in str X and processes it to tokens
    """
    #lowercases everything
    X = X.lower()
    
    #Removes all mentions and removes all punctuation
    subpattern = f'(@[A-z0-9]*)|[{string.punctuation[1:].replace("@","")}]*'
    replacer = re.compile(subpattern)
    X = replacer.sub('',X)
    
    #Tokenizes the text. wrote it this way so that it also pulls words with numbers
    tokenpattern = '([0-9]*[a-z]+[0-9]*[a-z]*)'
    tokenizer = re.compile(tokenpattern)
    X = tokenizer.findall(X)
    
    #Removes stopwords
    stopwords_list = stopwords.words('english') + ['sxsw']
    X = [word for word in X if word not in stopwords_list]
    
    #lemmatizes
    lemmatizer = WordNetLemmatizer()
    X = [lemmatizer.lemmatize(word) for word in X]
    X = ' '.join(X)
    
    return X
```

# EDA
After cleaning and tokenizing our data, I decided to explore our data and find the most frequent and impactful words by emotion and by products.
## N-grams
I looked at N-grams of length 2, 3, and 4 to see what were the most common phrases. Here I share just the Bigrams but feel free to look at them all within my [notebook](Notebooks/PreprocessingandEDA.ipynb).
### Positive Bigram

![](Images/bigramcutpos.png)
We can see here that the majority of the bigrams in the positive reviews are centered around the hype of the event, the location, and what each company was introducing.

### Negative Bigram

![](Images/bigramcutneg.png)
The negative bigrams are more centered on gripes with the company itself, design concerns people have with each company's products as well as at some of the new products that they introduced at the event.

## Word Clouds
Next I wanted to create word clouds for both positive and negative tweets about each company. Word clouds function in that the larger the word appears in the cloud, the more present and impactful it was in the reviews. For this, I also removed a bunch of additional stopwords for each product to reduce the number of common words within the positive and negative reviews of each product.By knowing the sentiment, the words that appear in each cloud can give us strong hints at what is and isn't working without having to read every tweet.

### Apple
#### Positive Tweets

![](Images/appleposcloud.png)
This is interesting! A lot of the positive talk was centered around the event. in removed words, their products appear very often in the positive reviews. Using a wordcloud like this, we can mark that the event was a success with the attendees. A lot of people enjoyed the popup shop and thought very positively about the new products that were shown within it.
#### Negative Tweets

![](Images/applenegcloud.png)
In the negative cloud, a lot of the criticsm their products faced had to do with aspects of their design, such as battery life. It appears that some people did not find the event as much of a success, and in fact found it quite painful. The words for the phrase 'fascist company' from our bigrams also appear here, so we know that some of the negativity was targeted at the company itself and not just the products.

### Google
If people were tweeting about google, they were talking about the new social media platform they were revealing at the event called Google Circles. both positive and negative tweets were about this.
#### Positive Tweets

![](Images/googleposcloud.png)
Of the positive things people had to say about Google Circles, it focused on the ability to connect people just like other social media platforms. Positive words such as 'excited', 'fun', 'great', and 'awesome' also appear here and show the type of positive feelings that were attached to this product.
#### Negative Tweets

![](Images/googlenegcloud.png)
The negative cloud is much more direct in expressing it's sentiment towards the product. Specifically, the word 'product' is huge, meaning people had issues with the product of Google Circles itself. Words such as 'suck', 'lost', and 'fail' give us a sense of the distaste or displeasure that people felt towards it.

# Modeling
## Preparing for Modeling
After tokenizing and vectorizing all words within the dataset, I performed Latent Semantic Analysis (LSA) to reduce the dimensionality of my data. I decided to truncate down to a dataset that still contained 98% of the explained variance.

![](Images/truncate.png)
I was able to reduce my dataset from 8,344 features down to 3,630 and still keep 98% of the explained variance.

## Modeling and Tuning
I split my data into target and test sets and ran a Logistic Regression, Random Forest, XGBooster, and a Support Vector Machine classifier on the original data. Afterwards, I performed SMOTE on the data to better deal with the class imbalance and then tuned the models using GridSearchCV. For scoring, I decided it would be best to use the macro f1 score for a few reasons: Macro f1 score tells us the strength of both our precision and recall scores, due to class imbalance accuracy would not tell us what we need to know about the classification of positive and negative tweets because of the overwhelming amount of tweets with no emotion, and I care more about correctly classifying positive and negative tweets than overall accuracy and the macro f1 score can give us a better sense of that.

![](Images/roccurvesvc.png)
This shows the roc curve for my SVC classifier. One reason for tuning our classifiers was to attempt to reduce overfitting and better improve classification of the negative tweets

## Scores
|Classifier| Initial Macro F1 Score | Final Macro F1 Score|Initial Accuracy |Final Accuracy|
|:---:| :---:| :---:| :---: |:---:|
|RandomForestClassifier| 0.59| 0.73|0.78| 0.85|
|XGBClassifier| 0.63| 0.75|0.88| 0.89|
|LogisticRegressionCV| 0.75 |0.77 |0.90| 0.89|
|SVC| 0.66|0.78|0.89|0.90|
# Conclusion
For all models, we had an improvement in the macro f1 score, which is good! It means my classifiers were able to better classify the minority classes. Random Forest was the model that most benefitted from tuning and SMOTE. The Support Vector Machines model had the best scores in terms of macro f1 scores and accuracy and thus my best functioning model although all models are fairly strong here.
# Final Remarks and Future Work
Twitter sentiment analysis can be a powerful tool for helping companies and brands recognize what people like and dislike about the brand or of the products. It is importatnt to note that while many people are on Twitter, not every customer uses an account, nor do they actively state their opinions all the time. Twitter data is only a subset of the whole but it is still helpful in painting a picture. In future work I'd like to revisit my method of preprocessing and see if there are different ways to use regular expressions to more cleanly extract the data. I would also want to stack my models to see if a stacked approach would give me a better overall score for both macro f1 and accuracy.
