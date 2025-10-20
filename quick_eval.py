import os
import time
import warnings
warnings.filterwarnings('ignore')

# CRITICAL: Set these BEFORE importing any other libraries
os.environ["USE_TF"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("🚀 QUICK EVALUATION - Focus on Best Models Only")
print("="*60)

# Load data
print("Loading data...")
df = pd.read_parquet("project_c_samples_3k.parquet")
print(f"Loaded {len(df)} records")

# Use smaller sample for quick evaluation
df = df.sample(n=1000, random_state=42)  # 1000 samples instead of 3000
print(f"Using {len(df)} samples for quick evaluation")

# Create ground truth (same logic as before)
def create_ground_truth(row):
    try:
        name_a = json.loads(row['names'])['primary'] if pd.notna(row['names']) else ""
        name_b = json.loads(row['base_names'])['primary'] if pd.notna(row['base_names']) else ""
        
        addr_a = ""
        addr_b = ""
        if pd.notna(row['addresses']):
            addr_data = json.loads(row['addresses'])
            if isinstance(addr_data, list) and len(addr_data) > 0:
                addr_a = addr_data[0].get('freeform', '')
        if pd.notna(row['base_addresses']):
            addr_data = json.loads(row['base_addresses'])
            if isinstance(addr_data, list) and len(addr_data) > 0:
                addr_b = addr_data[0].get('freeform', '')
        
        name_match = (
            name_a.lower() == name_b.lower() or
            name_a.lower() in name_b.lower() or
            name_b.lower() in name_a.lower()
        )
        
        addr_match = (
            addr_a.lower() == addr_b.lower() or
            (addr_a and addr_b and (
                any(word in addr_b.lower() for word in addr_a.lower().split() if len(word) > 3) or
                any(word in addr_a.lower() for word in addr_b.lower().split() if len(word) > 3)
            ))
        )
        
        cat_match = False
        if pd.notna(row['categories']) and pd.notna(row['base_categories']):
            try:
                cat_a = json.loads(row['categories'])
                cat_b = json.loads(row['base_categories'])
                if isinstance(cat_a, dict) and isinstance(cat_b, dict):
                    primary_a = cat_a.get('primary', '')
                    primary_b = cat_b.get('primary', '')
                    cat_match = primary_a == primary_b
            except:
                pass
        
        is_match = name_match and (addr_match or cat_match)
        return 1 if is_match else 0
    except Exception:
        return 0

df['ground_truth'] = df.apply(create_ground_truth, axis=1)
print(f"Ground truth distribution: {df['ground_truth'].value_counts().to_dict()}")

# Extract and normalize text
def extract_name(cell):
    if pd.isna(cell):
        return ""
    try:
        d = json.loads(cell) if isinstance(cell, str) else cell
        if isinstance(d, dict) and "primary" in d:
            return d["primary"]
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

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["name_a"] = df["names"].apply(extract_name)
df["name_b"] = df["base_names"].apply(extract_name)
df["addr_a"] = df["addresses"].apply(extract_address)
df["addr_b"] = df["base_addresses"].apply(extract_address)

# Enhanced text with normalization
df["text_a"] = df["name_a"].apply(normalize_text) + " " + df["addr_a"].apply(normalize_text)
df["text_b"] = df["name_b"].apply(normalize_text) + " " + df["addr_b"].apply(normalize_text)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth'])
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Test only the most promising models
models_to_test = [
    ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
    ('paraphrase-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'),
    ('all-mpnet-base-v2', 'all-mpnet-base-v2')  # Only top 3
]

def quick_evaluate(model_name, model_path):
    print(f"\n🔍 Testing {model_name}...")
    
    try:
        model = SentenceTransformer(model_path, device='cpu')
        
        # Quick threshold optimization
        print("  Optimizing threshold...")
        emb_a = model.encode(train_df["text_a"].tolist(), batch_size=32, show_progress_bar=False)
        emb_b = model.encode(train_df["text_b"].tolist(), batch_size=32, show_progress_bar=False)
        
        cosine_scores = np.array([
            cosine_similarity([emb_a[i]], [emb_b[i]])[0][0]
            for i in range(len(emb_a))
        ])
        
        ground_truth = train_df['ground_truth'].values
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 1.0, 0.1):
            predictions = (cosine_scores > threshold).astype(int)
            f1 = f1_score(ground_truth, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Test on test set
        print("  Evaluating on test set...")
        start_time = time.time()
        
        emb_a_test = model.encode(test_df["text_a"].tolist(), batch_size=32, show_progress_bar=False)
        emb_b_test = model.encode(test_df["text_b"].tolist(), batch_size=32, show_progress_bar=False)
        
        cosine_scores_test = np.array([
            cosine_similarity([emb_a_test[i]], [emb_b_test[i]])[0][0]
            for i in range(len(emb_a_test))
        ])
        
        predictions = (cosine_scores_test > best_threshold).astype(int)
        ground_truth_test = test_df['ground_truth'].values
        
        precision = precision_score(ground_truth_test, predictions)
        recall = recall_score(ground_truth_test, predictions)
        f1 = f1_score(ground_truth_test, predictions)
        
        total_time = time.time() - start_time
        time_per_match = (total_time / len(test_df)) * 1000
        
        print(f"  ✅ Results: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        print(f"  ⏱️  Speed: {time_per_match:.1f}ms per match")
        print(f"  🎯 Threshold: {best_threshold:.2f}")
        
        # Check OKRs
        meets_f1 = f1 >= 0.90
        meets_speed = time_per_match <= 50
        
        print(f"  📊 OKR Status:")
        print(f"     F1 ≥ 90%: {'✅ YES' if meets_f1 else '❌ NO'} ({f1:.1%})")
        print(f"     Speed ≤ 50ms: {'✅ YES' if meets_speed else '❌ NO'} ({time_per_match:.1f}ms)")
        print(f"     Both OKRs: {'🎉 ACHIEVED!' if meets_f1 and meets_speed else '❌ Not yet'}")
        
        return {
            'model': model_name,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'speed': time_per_match,
            'threshold': best_threshold,
            'meets_f1_okr': meets_f1,
            'meets_speed_okr': meets_speed
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# Run quick evaluation
print(f"\n🚀 Starting quick evaluation of {len(models_to_test)} models...")
results = []

for model_name, model_path in models_to_test:
    result = quick_evaluate(model_name, model_path)
    if result:
        results.append(result)

# Summary
if results:
    print(f"\n🏆 QUICK EVALUATION SUMMARY")
    print("="*50)
    
    # Sort by F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['model']}")
        print(f"   F1: {result['f1']:.3f} | Speed: {result['speed']:.1f}ms | Threshold: {result['threshold']:.2f}")
        print(f"   OKRs: F1={'✅' if result['meets_f1_okr'] else '❌'} Speed={'✅' if result['meets_speed_okr'] else '❌'}")
        print()
    
    # Check if any model meets both OKRs
    okr_models = [r for r in results if r['meets_f1_okr'] and r['meets_speed_okr']]
    
    if okr_models:
        best = okr_models[0]
        print(f"🎉 SUCCESS! {best['model']} meets both OKRs!")
        print(f"   F1: {best['f1']:.1%} (target: 90%)")
        print(f"   Speed: {best['speed']:.1f}ms (target: ≤50ms)")
    else:
        best = results[0]
        print(f"⚠️  No model meets both OKRs yet.")
        print(f"   Best F1: {best['f1']:.1%} (need {90-best['f1']*100:.1f}% more)")
        print(f"   Best Speed: {min(r['speed'] for r in results):.1f}ms ✅")
        
        print(f"\n💡 Next steps to reach 90% F1:")
        print(f"   1. Try larger models (RoBERTa, BERT-large)")
        print(f"   2. Ensemble multiple models")
        print(f"   3. Fine-tune on place conflation data")
        print(f"   4. Add more sophisticated text preprocessing")

print(f"\n⏱️  Total evaluation time: {time.time() - start_time:.1f} seconds")
