import os

# CRITICAL: Set these BEFORE importing any other libraries
os.environ["USE_TF"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # This is key!

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

print("Loading data...")
df = pd.read_parquet("project_c_samples_3k.parquet")
df = df.head(100)

def extract_name(cell):
    if pd.isna(cell):
        return ""
    try:
        d = json.loads(cell) if isinstance(cell, str) else cell
        if isinstance(d, dict) and "primary" in d:
            return d["primary"]
        if isinstance(d, dict) and "names" in d:
            names = d["names"]
            return next(iter(names.values())) if names else ""
        return ""
    except Exception:
        return ""
    
def extract_address(cell):
    if pd.isna(cell):
        return ""
    try:
        arr = json.loads(cell) if isinstance(cell, str) else cell
        if isinstance(arr, list) and len(arr) > 0 and "freeform" in arr[0]:
            return arr[0]["freeform"]
        return ""
    except Exception:
        return ""

print("Extracting fields...")
df["name_a"] = df["names"].apply(extract_name)
df["name_b"] = df["base_names"].apply(extract_name)
df["addr_a"] = df["addresses"].apply(extract_address)
df["addr_b"] = df["base_addresses"].apply(extract_address)

df["text_a"] = df["name_a"].fillna('') + " " + df["addr_a"].fillna('')
df["text_b"] = df["name_b"].fillna('') + " " + df["addr_b"].fillna('')

print("\nSample data:")
print(df[["name_a", "name_b", "addr_a", "addr_b"]].head(3))

print("\nLoading model (first run may take 1-2 minutes)...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Model loaded!")

print("\nEncoding text_a...")
emb_a = model.encode(
    df["text_a"].tolist(),
    batch_size=4,  # Even smaller batch
    convert_to_tensor=False,
    show_progress_bar=True,
    device='cpu'
)
print(f"✓ Encoded {len(emb_a)} embeddings")

print("\nEncoding text_b...")
emb_b = model.encode(
    df["text_b"].tolist(),
    batch_size=4,
    convert_to_tensor=False,
    show_progress_bar=True,
    device='cpu'
)
print(f"✓ Encoded {len(emb_b)} embeddings")

print("\nCalculating similarities...")
# Simpler cosine similarity calculation
cosine_scores = np.array([
    cosine_similarity([emb_a[i]], [emb_b[i]])[0][0]
    for i in range(len(emb_a))
])

threshold = 0.8
df["predicted_match"] = cosine_scores > threshold
df["similarity_score"] = cosine_scores

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(df[["name_a", "name_b", "similarity_score", "predicted_match"]].head(20))
print(f"\nPredicted matches: {df['predicted_match'].sum()} out of {len(df)}")

# Distribution of scores
print(f"\nSimilarity score distribution:")
print(f"  Mean: {cosine_scores.mean():.3f}")
print(f"  Min: {cosine_scores.min():.3f}")
print(f"  Max: {cosine_scores.max():.3f}")
print(f"  Std: {cosine_scores.std():.3f}")

# # Add this to your existing script:

# print("\n" + "="*80)
# print("DETAILED ANALYSIS:")
# print("="*80)

# # Cases where embedding model disagrees with current matcher
# low_similarity = df[df['similarity_score'] < threshold].copy()
# print(f"\n🔴 LOW SIMILARITY MATCHES ({len(low_similarity)} cases)")
# print("Current matcher said YES, but embedding similarity < {:.2f}:".format(threshold))
# print(low_similarity[['name_a', 'name_b', 'addr_a', 'addr_b', 'similarity_score']].head(10))

# # Perfect matches
# perfect = df[df['similarity_score'] > 0.95].copy()
# print(f"\n✅ HIGH CONFIDENCE MATCHES ({len(perfect)} cases)")
# print("Similarity > 0.95:")
# print(perfect[['name_a', 'name_b', 'similarity_score']].head(10))

# # Borderline cases
# borderline = df[(df['similarity_score'] >= 0.75) & (df['similarity_score'] <= 0.85)].copy()
# print(f"\n⚠️  BORDERLINE CASES ({len(borderline)} cases)")
# print("Similarity between 0.75-0.85:")
# print(borderline[['name_a', 'name_b', 'addr_a', 'addr_b', 'similarity_score']].head(10))

# # Save results for manual review
# df.to_csv('model_predictions.csv', index=False)
# print("\n💾 Results saved to 'model_predictions.csv'")