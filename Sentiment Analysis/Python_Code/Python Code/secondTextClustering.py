import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import homogeneity_score,silhouette_score
import matplotlib.pyplot as plt
f = open("Sapir1921_chapter1.txt", "r")
text1=f.read()
print (text1)
stop_words=set(stopwords.words('english'))
text2=''.join(str(x)for x in text1)
documents=nltk.word_tokenize(text2)

wordcloud=WordCloud(stopwords=stop_words, background_color='white').generate(text1)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

vec = TfidfVectorizer(stop_words="english", use_idf=True)
vec.fit(documents)
features = vec.transform(documents)
true_k = 2
cls = MiniBatchKMeans(n_clusters=true_k, random_state=True).fit(features)
cls.labels_
cls.predict(features)
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
sse = []
for k in range(1, 11):
     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
     kmeans.fit(features)
     sse.append(kmeans.inertia_)
plt.plot(range(1,11),sse)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()
sorted_centroids = cls.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in sorted_centroids[i, :5]:
        print(' %s' % terms[ind], end='')
    print()
    print()
print()
h=homogeneity_score(documents, cls.predict(features))
s=silhouette_score(features, labels=cls.predict(features))
print(s,h)
