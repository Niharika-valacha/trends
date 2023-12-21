# Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import wordcloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import plotly.express as px

# Read datasets/papers.csv into papers
papers = pd.read_csv('datasets/papers.csv')

# Data Cleaning
papers.drop(['id', 'event_type', 'pdf_name'], axis=1, inplace=True)
papers.dropna(inplace=True)

# Explore the distribution of papers across years
sns.countplot(x='year', data=papers)
plt.title('Distribution of Papers Across Years')
plt.show()

# Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

papers['title_processed'] = papers['title'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in re.findall(r'\b\w+\b', x) if word.lower() not in stop_words]))

# Word Cloud
long_string = " ".join(papers['title_processed'])
wc = wordcloud.WordCloud(background_color='white', colormap='viridis', width=800, height=400)
wc.generate(long_string)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Processed Titles')
plt.show()

# Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(papers['title_processed'].values)

# Visualize the 10 most common words
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = list(zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.bar(x_pos, counts, align='center')
    plt.xticks(x_pos, words, rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('10 Most Common Words')
    plt.show()

plot_10_most_common_words(count_data, count_vectorizer)

# LDA Topic Modeling
number_topics = 10
number_words = 10

lda = LDA(n_components=number_topics)
lda.fit(count_data)

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx}:")
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

# Interactive Visualization with Plotly
fig = px.bar(papers['year'].value_counts().reset_index(), x='index', y='year', labels={'index': 'Year', 'year': 'Number of Papers'}, title='Number of Papers Published Each Year')
fig.show()
