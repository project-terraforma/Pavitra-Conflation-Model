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
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # This is key!

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_parquet("project_c_samples_3k.parquet")
# Use full dataset for better evaluation
print(f"Loaded {len(df)} records")

# Create ground truth labels based on multiple criteria
# This is a dataset of potential matches - we need to determine if they're the same entity
def create_ground_truth(row):
    """Create ground truth based on multiple similarity criteria"""
    try:
        # Extract names
        name_a = json.loads(row['names'])['primary'] if pd.notna(row['names']) else ""
        name_b = json.loads(row['base_names'])['primary'] if pd.notna(row['base_names']) else ""
        
        # Extract addresses
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
        
        # Criteria for match:
        # 1. Names are very similar (exact match or one contains the other)
        name_match = (
            name_a.lower() == name_b.lower() or
            name_a.lower() in name_b.lower() or
            name_b.lower() in name_a.lower()
        )
        
        # 2. Addresses are similar (same street/area)
        addr_match = (
            addr_a.lower() == addr_b.lower() or
            (addr_a and addr_b and (
                any(word in addr_b.lower() for word in addr_a.lower().split() if len(word) > 3) or
                any(word in addr_a.lower() for word in addr_b.lower().split() if len(word) > 3)
            ))
        )
        
        # 3. Categories are similar
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
        
        # Match if name matches AND (address matches OR category matches)
        is_match = name_match and (addr_match or cat_match)
        
        return 1 if is_match else 0
        
    except Exception as e:
        return 0

print("Creating ground truth labels...")
df['ground_truth'] = df.apply(create_ground_truth, axis=1)
print(f"Ground truth distribution: {df['ground_truth'].value_counts().to_dict()}")
print(f"Match rate: {df['ground_truth'].mean():.2%}")

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

def normalize_text(text):
    """Enhanced text normalization for better matching"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common business suffixes
    suffixes = ['inc', 'llc', 'corp', 'ltd', 'co', 'company', 'corporation', 
                'limited', 'incorporated', 'group', 'associates', 'partners']
    for suffix in suffixes:
        text = text.replace(f' {suffix}', '').replace(f' {suffix}.', '')
    
    # Remove punctuation and extra spaces
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle common abbreviations
    abbreviations = {
        'st': 'street', 'ave': 'avenue', 'rd': 'road', 'blvd': 'boulevard',
        'dr': 'drive', 'ct': 'court', 'ln': 'lane', 'pl': 'place',
        'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
        'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest'
    }
    
    words = text.split()
    normalized_words = []
    for word in words:
        if word in abbreviations:
            normalized_words.append(abbreviations[word])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)
    
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

# Apply text normalization
df["name_a_norm"] = df["name_a"].apply(normalize_text)
df["name_b_norm"] = df["name_b"].apply(normalize_text)
df["addr_a_norm"] = df["addr_a"].apply(normalize_text)
df["addr_b_norm"] = df["addr_b"].apply(normalize_text)

# Create enhanced text combinations
df["text_a"] = df["name_a_norm"].fillna('') + " " + df["addr_a_norm"].fillna('')
df["text_b"] = df["name_b_norm"].fillna('') + df["addr_b_norm"].fillna('')

# Add category information for better context
def extract_category(cell):
    if pd.isna(cell):
        return ""
    try:
        d = json.loads(cell) if isinstance(cell, str) else cell
        if isinstance(d, dict) and "primary" in d:
            return d["primary"]
        return ""
    except Exception:
        return ""

df["cat_a"] = df["categories"].apply(extract_category)
df["cat_b"] = df["base_categories"].apply(extract_category)

# Enhanced text with category context
df["text_a_enhanced"] = df["text_a"] + " " + df["cat_a"].fillna('')
df["text_b_enhanced"] = df["text_b"] + " " + df["cat_b"].fillna('')

print("\nSample data:")
print(df[["name_a", "name_b", "addr_a", "addr_b", "ground_truth"]].head(3))

# Split data for evaluation
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth'])
print(f"\nTrain set: {len(train_df)} records, Test set: {len(test_df)} records")

def evaluate_model(model_name, model, threshold=0.8, test_data=None):
    """Evaluate a model with timing and metrics"""
    if test_data is None:
        test_data = test_df
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Time the encoding process
    start_time = time.time()
    
    # Use enhanced text for better matching
    text_a_list = test_data["text_a_enhanced"].tolist() if "text_a_enhanced" in test_data.columns else test_data["text_a"].tolist()
    text_b_list = test_data["text_b_enhanced"].tolist() if "text_b_enhanced" in test_data.columns else test_data["text_b"].tolist()
    
    print("Encoding text_a...")
    emb_a = model.encode(
        text_a_list,
        batch_size=16,
        convert_to_tensor=False,
        show_progress_bar=True,
        device='cpu'
    )
    
    print("Encoding text_b...")
    emb_b = model.encode(
        text_b_list,
        batch_size=16,
        convert_to_tensor=False,
        show_progress_bar=True,
        device='cpu'
    )
    
    encoding_time = time.time() - start_time
    
    # Calculate similarities
    print("Calculating similarities...")
    similarity_start = time.time()
    cosine_scores = np.array([
        cosine_similarity([emb_a[i]], [emb_b[i]])[0][0]
        for i in range(len(emb_a))
    ])
    similarity_time = time.time() - similarity_start
    
    # Make predictions
    predictions = (cosine_scores > threshold).astype(int)
    ground_truth = test_data['ground_truth'].values
    
    # Calculate metrics
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    
    # Calculate timing per match
    total_time = encoding_time + similarity_time
    time_per_match = (total_time / len(test_data)) * 1000  # Convert to ms
    
    results = {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'encoding_time': encoding_time,
        'similarity_time': similarity_time,
        'total_time': total_time,
        'time_per_match_ms': time_per_match,
        'threshold': threshold,
        'n_samples': len(test_data)
    }
    
    print(f"\nRESULTS:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Encoding time: {encoding_time:.2f}s")
    print(f"Similarity time: {similarity_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per match: {time_per_match:.2f}ms")
    print(f"Threshold: {threshold}")
    
    return results, cosine_scores, predictions

def optimize_threshold(model, test_data=None):
    """Find optimal threshold for maximum F1 score"""
    if test_data is None:
        test_data = test_df
    
    print(f"\nOptimizing threshold for maximum F1 score...")
    
    # Get embeddings with enhanced text
    text_a_list = test_data["text_a_enhanced"].tolist() if "text_a_enhanced" in test_data.columns else test_data["text_a"].tolist()
    text_b_list = test_data["text_b_enhanced"].tolist() if "text_b_enhanced" in test_data.columns else test_data["text_b"].tolist()
    
    emb_a = model.encode(text_a_list, batch_size=16, show_progress_bar=False)
    emb_b = model.encode(text_b_list, batch_size=16, show_progress_bar=False)
    
    cosine_scores = np.array([
        cosine_similarity([emb_a[i]], [emb_b[i]])[0][0]
        for i in range(len(emb_a))
    ])
    
    ground_truth = test_data['ground_truth'].values
    
    # Test different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (cosine_scores > threshold).astype(int)
        f1 = f1_score(ground_truth, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1

def ensemble_prediction(models_results, test_data):
    """Create ensemble prediction from multiple models"""
    print("\nCreating ensemble prediction...")
    
    # Get predictions from top 3 models
    top_models = sorted(models_results, key=lambda x: x['f1_score'], reverse=True)[:3]
    
    ensemble_scores = []
    for i in range(len(test_data)):
        # Weighted average of top 3 models
        weighted_score = 0
        total_weight = 0
        
        for model_result in top_models:
            weight = model_result['f1_score']  # Use F1 as weight
            # Get similarity score for this model (would need to store these)
            # For now, we'll use a simplified approach
            weighted_score += model_result['f1_score'] * 0.8  # Placeholder
            total_weight += weight
        
        ensemble_scores.append(weighted_score / total_weight if total_weight > 0 else 0.5)
    
    return np.array(ensemble_scores)

def calculate_cost_analysis(results_df):
    """Calculate cost analysis for price-to-performance ratio"""
    # Model size estimates (in MB) - approximate values
    model_sizes = {
        'all-MiniLM-L6-v2': 22.7,  # ~23MB
        'all-mpnet-base-v2': 420,   # ~420MB  
        'paraphrase-MiniLM-L6-v2': 22.7,  # ~23MB
        'distilbert-base-nli-mean-tokens': 250  # ~250MB
    }
    
    # Estimated inference cost per 1M tokens (relative pricing)
    # These are rough estimates based on model complexity
    cost_per_1m_tokens = {
        'all-MiniLM-L6-v2': 0.10,      # Cheapest, smallest
        'all-mpnet-base-v2': 0.50,      # Most expensive, largest
        'paraphrase-MiniLM-L6-v2': 0.12, # Slightly more than MiniLM
        'distilbert-base-nli-mean-tokens': 0.30  # Medium cost
    }
    
    # Calculate cost metrics
    results_df['model_size_mb'] = results_df['model_name'].map(model_sizes)
    results_df['cost_per_1m_tokens'] = results_df['model_name'].map(cost_per_1m_tokens)
    
    # Calculate performance per dollar (F1 score per cost unit)
    results_df['performance_per_dollar'] = results_df['f1_score'] / results_df['cost_per_1m_tokens']
    
    # Calculate efficiency score (F1 per MB of model size)
    results_df['efficiency_score'] = results_df['f1_score'] / results_df['model_size_mb']
    
    # Calculate overall price-to-performance ratio
    results_df['price_performance_ratio'] = (
        results_df['f1_score'] / 
        (results_df['cost_per_1m_tokens'] * results_df['model_size_mb'] / 100)
    )
    
    return results_df

# Define models to test - including more powerful options
models_to_test = [
    ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
    ('all-mpnet-base-v2', 'all-mpnet-base-v2'),
    ('paraphrase-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'),
    ('distilbert-base-nli-mean-tokens', 'distilbert-base-nli-mean-tokens'),
    ('sentence-transformers/all-roberta-large-v1', 'sentence-transformers/all-roberta-large-v1'),
    ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
]

all_results = []

print("\n" + "="*80)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*80)

for model_name, model_path in models_to_test:
    try:
        print(f"\nLoading {model_name}...")
        model = SentenceTransformer(model_path, device='cpu')
        
        # Optimize threshold for this model
        optimal_threshold, optimal_f1 = optimize_threshold(model, train_df)
        
        # Evaluate with optimal threshold
        results, scores, predictions = evaluate_model(
            model_name, model, optimal_threshold, test_df
        )
        
        all_results.append(results)
        
        # Check if this model meets our OKRs
        meets_f1_okr = results['f1_score'] >= 0.90
        meets_speed_okr = results['time_per_match_ms'] <= 50
        
        print(f"\nOKR STATUS:")
        print(f"✓ F1 Score ≥ 90%: {'YES' if meets_f1_okr else 'NO'} ({results['f1_score']:.1%})")
        print(f"✓ Speed ≤ 50ms: {'YES' if meets_speed_okr else 'NO'} ({results['time_per_match_ms']:.1f}ms)")
        print(f"✓ Both OKRs met: {'YES' if meets_f1_okr and meets_speed_okr else 'NO'}")
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        continue

# Create results summary
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Add cost analysis
    results_df = calculate_cost_analysis(results_df)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by F1 score
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\nModel Performance Ranking (by F1 Score):")
    for i, row in results_df.iterrows():
        print(f"{row['model_name']:30} | F1: {row['f1_score']:.3f} | Speed: {row['time_per_match_ms']:6.1f}ms | Threshold: {row['threshold']:.3f}")
    
    # Price-to-performance analysis
    print(f"\n" + "="*60)
    print("PRICE-TO-PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Sort by price-performance ratio
    results_df_cost = results_df.sort_values('price_performance_ratio', ascending=False)
    
    print("\nBest Price-to-Performance Models:")
    for i, row in results_df_cost.iterrows():
        print(f"{row['model_name']:30} | F1: {row['f1_score']:.3f} | Cost: ${row['cost_per_1m_tokens']:.2f}/1M | Size: {row['model_size_mb']:4.0f}MB | Ratio: {row['price_performance_ratio']:.2f}")
    
    # Find best overall model
    best_overall = results_df_cost.iloc[0]
    print(f"\n🏆 BEST PRICE-TO-PERFORMANCE MODEL:")
    print(f"Model: {best_overall['model_name']}")
    print(f"F1 Score: {best_overall['f1_score']:.3f}")
    print(f"Speed: {best_overall['time_per_match_ms']:.1f}ms per match")
    print(f"Cost: ${best_overall['cost_per_1m_tokens']:.2f} per 1M tokens")
    print(f"Model Size: {best_overall['model_size_mb']:.0f}MB")
    print(f"Price-Performance Ratio: {best_overall['price_performance_ratio']:.2f}")
    
    # Find best model meeting both OKRs
    okr_models = results_df[(results_df['f1_score'] >= 0.90) & (results_df['time_per_match_ms'] <= 50)]
    
    if len(okr_models) > 0:
        best_model = okr_models.iloc[0]
        print(f"\n🏆 BEST MODEL MEETING BOTH OKRs:")
        print(f"Model: {best_model['model_name']}")
        print(f"F1 Score: {best_model['f1_score']:.3f}")
        print(f"Speed: {best_model['time_per_match_ms']:.1f}ms per match")
        print(f"Optimal Threshold: {best_model['threshold']:.3f}")
    else:
        print(f"\n⚠️  NO MODEL MEETS BOTH OKRs")
        print("Models meeting F1 ≥ 90%:")
        f1_models = results_df[results_df['f1_score'] >= 0.90]
        if len(f1_models) > 0:
            for _, row in f1_models.iterrows():
                print(f"  {row['model_name']}: F1={row['f1_score']:.3f}, Speed={row['time_per_match_ms']:.1f}ms")
        else:
            print("  None")
            
        print("Models meeting Speed ≤ 50ms:")
        speed_models = results_df[results_df['time_per_match_ms'] <= 50]
        if len(speed_models) > 0:
            for _, row in speed_models.iterrows():
                print(f"  {row['model_name']}: F1={row['f1_score']:.3f}, Speed={row['time_per_match_ms']:.1f}ms")
        else:
            print("  None")
    
    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nResults saved to 'model_comparison_results.csv'")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # F1 Score comparison
    axes[0, 0].bar(results_df['model_name'], results_df['f1_score'])
    axes[0, 0].axhline(y=0.90, color='r', linestyle='--', label='OKR Target (90%)')
    axes[0, 0].set_title('F1 Score Comparison')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    
    # Speed comparison
    axes[0, 1].bar(results_df['model_name'], results_df['time_per_match_ms'])
    axes[0, 1].axhline(y=50, color='r', linestyle='--', label='OKR Target (50ms)')
    axes[0, 1].set_title('Speed Comparison')
    axes[0, 1].set_ylabel('Time per Match (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    
    # Performance vs Speed trade-off
    scatter = axes[0, 2].scatter(results_df['time_per_match_ms'], results_df['f1_score'], 
                                s=results_df['model_size_mb']*2, alpha=0.7, c=results_df['cost_per_1m_tokens'], 
                                cmap='viridis')
    for i, row in results_df.iterrows():
        axes[0, 2].annotate(row['model_name'], (row['time_per_match_ms'], row['f1_score']), 
                           fontsize=8, ha='center')
    axes[0, 2].axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Time per Match (ms)')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Performance vs Speed (size=model_size, color=cost)')
    plt.colorbar(scatter, ax=axes[0, 2], label='Cost per 1M tokens ($)')
    
    # Price-to-performance ratio
    axes[1, 0].bar(results_df['model_name'], results_df['price_performance_ratio'])
    axes[1, 0].set_title('Price-to-Performance Ratio')
    axes[1, 0].set_ylabel('F1 Score / (Cost × Size)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Model size comparison
    axes[1, 1].bar(results_df['model_name'], results_df['model_size_mb'])
    axes[1, 1].set_title('Model Size Comparison')
    axes[1, 1].set_ylabel('Model Size (MB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Cost comparison
    axes[1, 2].bar(results_df['model_name'], results_df['cost_per_1m_tokens'])
    axes[1, 2].set_title('Cost per 1M Tokens')
    axes[1, 2].set_ylabel('Cost ($)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive visualization saved to 'model_comparison_analysis.png'")
    
else:
    print("No models were successfully evaluated.")

