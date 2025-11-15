from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "Computer Science degree"
text2 = "BSc in Computer Science"

embeddings = model.encode([text1, text2])
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print(f"Cosine similarity between '{text1}' and '{text2}': {similarity:.4f}")
