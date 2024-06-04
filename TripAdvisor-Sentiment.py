import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

df = pd.read_csv("dataset/tripadvisor_hotel_reviews.csv")
df.head()
df.info()


###############################
#     TEXT PREPROCESSING      #
###############################
def prepocessing(dataframe, column_name, rare_count):
    """

    :param dataframe: dataframe to preprocess e.g.(df)
    :param column_name: column of data to preprocess e.g.(review,overview)
    :param rare_count: The number of observations threshold for removing a small number of observed words e.g.(500)
    :return:preprocessed data
    """
    dataframe[f"{column_name}"] = dataframe[f"{column_name}"].str.lower()  # Normalizing Case Folding
    dataframe[f"{column_name}"] = dataframe[f"{column_name}"].str.replace('[^\w\s]', '', regex=True)  # Punctuations
    dataframe[f"{column_name}"] = dataframe[f"{column_name}"].str.replace('\d', '', regex=True)  # Numbers

    # Stopwords #
    sw = stopwords.words('english')
    dataframe[f"{column_name}"] = (dataframe[f"{column_name}"]
                                   .apply(lambda x: " ".join(x for x in str(x).split() if x not in sw)))

    # Rarewords #
    rw = pd.Series(' '.join(dataframe[f"{column_name}"]).split()).value_counts()[-rare_count:]
    dataframe[f"{column_name}"] = (dataframe[f"{column_name}"]
                                   .apply(lambda x: " ".join(x for x in x.split() if x not in rw)))

    # Lemmatization #
    dataframe[f"{column_name}"] = (dataframe[f"{column_name}"]
                                   .apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])))

    return dataframe[f"{column_name}"]


df['Review'] = prepocessing(df, 'Review', 1000)


##############################################################
#                    Text Visualization                      #
##############################################################


#  Bar plot

def barplot(dataframe, col_name, frequency):
    tf = dataframe[col_name].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]

    # Filter words with frequency more than "frequency"
    filtered_tf = tf[tf["tf"] > frequency]

    # Coloring the bar chart
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set1", len(filtered_tf))

    # Create a bar chart
    bars = plt.bar(filtered_tf["words"], filtered_tf["tf"], color=colors)

    # Show a color bar chart
    plt.xlabel('Words')
    plt.ylabel('Term Frequency (TF)')
    plt.title('Words with Term Frequency Greater Than 10000')
    plt.xticks(rotation=45)
    plt.show()


barplot(df, "Review", 10000)


##############################################################
#                         WORDCLOUD                          #
##############################################################
def wordcloudShow(dataframe, col_name):
    text = " ".join(i for i in dataframe[col_name])

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


wordcloudShow(df, "Review")


##############################################################
#                    Sentiment Analysis                      #
##############################################################

def sentimentLabelCreate(dataframe, col_name, new_col_name):
    sia = SentimentIntensityAnalyzer()
    dataframe[f"{new_col_name}"] = (dataframe[f"{col_name}"]
                                    .apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg"))

    return dataframe[f"{new_col_name}"]


sentimentLabelCreate(df, "Review", "Sentiment_Label")

# sentiment label averages according to ratings
df.groupby("Sentiment_Label")["Rating"].mean()
# Sentiment_Label
# neg   1.58
# pos   4.09


##############################################################
#             Preparation for Machine Learning               #
##############################################################

# Train-Test Split
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# TF-IDF Word Level Transform Data

tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


##############################################################
#                    LogisticRegression                      #
##############################################################


log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Forecast results
y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

#               precision    recall  f1-score   support
#          neg       0.18      0.96      0.31        51
#          pos       1.00      0.96      0.98      5072
#     accuracy                           0.96      5123
#    macro avg       0.59      0.96      0.64      5123
# weighted avg       0.99      0.96      0.97      5123



cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()
# 0.9482722942073171

# Selecting a random observation
random_review = pd.Series(df["Review"].sample(1).values)
# open mind fantastic time review little late nt...

# Transform
new_comment = CountVectorizer().fit(train_x).transform(random_review)

# <1x59211 sparse matrix of type '<class 'numpy.int64'>'
# with 143 stored elements in Compressed Sparse Row format>

pred = log_model.predict(new_comment)
# array(['pos'], dtype=object)
# print(f'Review:  {random_review[0]} \n Prediction: {pred}')
