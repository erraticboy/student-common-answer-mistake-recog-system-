from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Example student answers
answers = [
    "The capital of France is London",
    "Paris is the capital of France",
    "France capital is Paris",
    "London is the capital of France",
    "Paris is France's capital"
]

# Step 1: Vectorize text answers
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(answers)

# Step 2: Cluster answers to find common mistakes
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

clusters = kmeans.labels_

# Step 3: Summarize mistakes
cluster_summary = Counter(clusters)
print("Cluster distribution:", cluster_summary)

# Step 4: Draft feedback
for cluster_id in set(clusters):
    cluster_answers = [ans for ans, cid in zip(answers, clusters) if cid == cluster_id]
    print(f"\nCluster {cluster_id} examples:")
    for ans in cluster_answers:
        print("-", ans)

    if "London" in " ".join(cluster_answers):
        print("⚠️ Common mistake: Students confused London with Paris.")
    else:
        print("✅ Correct understanding: Paris is the capital of France.")