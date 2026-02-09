import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Streamlit UI
st.title("Teacher Analytics Dashboard")
st.write("Upload student answers and get insights!")

# Input box for answers
answers_input = st.text_area("Enter student answers (one per line):")
if answers_input:
    answers = answers_input.split("\n")

    # Vectorize
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(answers)

    # Cluster
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_

    # Show results
    st.subheader("Cluster Distribution")
    cluster_summary = Counter(clusters)
    st.write(cluster_summary)

    for cluster_id in set(clusters):
        st.subheader(f"Cluster {cluster_id}")
        cluster_answers = [ans for ans, cid in zip(answers, clusters) if cid == cluster_id]
        st.write(cluster_answers)

        if "London" in " ".join(cluster_answers):
            st.warning("⚠️ Common mistake: Students confused London with Paris.")
        else:
            st.success("✅ Correct understanding: Paris is the capital of France.")