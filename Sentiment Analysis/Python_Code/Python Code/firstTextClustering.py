#import important libraries
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# data collection
x =  ["This is a Cat", "I am happy with my friend", "I love ice-cream","He left his car outside the yard",
      "Python is challenging to understand", "Here is my bar code 1234ggr3666", "here is my address vaxjo 211843"]
#count vectorization
cv = CountVectorizer(analyzer = 'word', max_features = 5000, lowercase=True, preprocessor=None, tokenizer=None, stop_words = 'english')
vectors = cv.fit_transform(x)

#Kmeans clustering
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 0)
kmean_indices = kmeans.fit_predict(vectors)

#Dimensionality reduction
pca = PCA(n_components=6)

#Visualization
scatter_plot_points = pca.fit_transform(vectors.toarray())
colors = ["r", "b", "c", "y", "m","g" ]
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])
for i, txt in enumerate(x):
    ax.annotate(txt, (x_axis[i], y_axis[i]))
plt.show()