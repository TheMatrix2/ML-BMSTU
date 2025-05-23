#%%
import wikipedia
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk
import spacy
import certifi
import ssl
import string

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['SSL_CERT_FILE'] = certifi.where()

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
punctuation = set(string.punctuation)

wikipedia.set_lang("ru")
stop_words = set(stopwords.words('russian'))
#%% download articles
all_articles = []
all_titles = []
topics = ['Образование', 'Космические исследование', 'Экономика']
num_articles = 15
for topic in topics:
    print(f"Downloading articles about: {topic}")
    articles = []
    titles = []

    search_results = wikipedia.search(topic, results=num_articles*2)

    count = 0
    for title in search_results:
        if count >= num_articles:
            break

        try:
            page = wikipedia.page(title)
            articles.append(page.content)
            titles.append(title)
            count += 1
            print(f"Downloaded article: {title}")
        except Exception as e:
            print(f"Error downloading article '{title}': {e}")

    all_articles.append(articles)
    all_titles.append(titles)
#%%
articles = [article for sublist in all_articles for article in sublist]
titles = [title for sublist in all_titles for title in sublist]
#%% text preprocessing (stemming or lemmatization)
nlp = spacy.load('ru_core_news_sm')

def preprocess_text(text):
    # text = nlp(text.lower())
    # lemmatized_words = [token.lemma_ for token in text if token.text not in punctuation and token.text not in stop_words]
    # return " ".join(lemmatized_words)
    stemmer = SnowballStemmer('russian')
    tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_words)

vectorizer = TfidfVectorizer(preprocessor=preprocess_text, max_features=10000)
X = vectorizer.fit_transform(articles)
feature_names = vectorizer.get_feature_names_out()
print(vectorizer.get_feature_names_out())
print(X.shape)
print(feature_names)
#%% Elbow method
sum_of_squared_distances = []
K = range(1, 7)
for k in K:
    km = KMeans(n_clusters=k, max_iter=4000)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)
plt.plot(K, sum_of_squared_distances, 'rx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method')
plt.show()

#%% K-means
true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1500)
model.fit(X)
labels=model.labels_
wiki_cl=pd.DataFrame(list(zip(titles,labels)),columns=['title','cluster'])
print(wiki_cl.sort_values(by=['cluster']))
#%% Dendogram
mergins = linkage(X.toarray(), method='complete')
dendrogram(mergins,
           labels=titles,
           leaf_font_size=6,
           orientation='left',
           count_sort='descending',
           show_leaf_counts=True
           )

plt.show()
#%% DBSCAN
for eps in np.arange(0.1, 10, 0.05):
    for min_samples in range(3, 15):
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 3:
            print(f"eps={eps:.2f}, min_samples={min_samples}, clusters={n_clusters}")
            if n_clusters > 1:
                score = silhouette_score(X, labels)
                print(f"Silhouette score: {score:.4f}")
#%%
dbscan = DBSCAN(eps=0.45, min_samples=7, metric='cosine')
dbscan.fit(X)

labels = dbscan.labels_
wiki_cl = pd.DataFrame(list(zip(articles, labels)), columns=['article', 'cluster'])
wiki_cl_sorted = wiki_cl.sort_values(by='cluster')

print(wiki_cl_sorted)
#%% vizualization
import umap

umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=30)
plt.title("UMAP Visualization of TF-IDF Vectors")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()