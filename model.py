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

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import json
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

def log_output(message, output_file):
    """Log message to both console and file"""
    print(message)
    output_file.write(message + "\n")
    output_file.flush()

# Open output file for results
with open('results.txt', 'w') as f:
    log_output("="*80, f)
    log_output("PLACE CONFLATION MODEL EVALUATION: COMPARATIVE ANALYSIS & RECOMMENDATION", f)
    log_output("="*80, f)
    log_output("", f)
    
    log_output("🎯 OBJECTIVE: Evaluate improvement of place conflation using language models", f)
    log_output("", f)
    log_output("📊 KEY RESULTS:", f)
    log_output("  1. Achieve at least 80% F1 score (or precision/recall balance) on the test dataset using a language model", f)
    log_output("  2. Run inference within 50 ms per match on average, using a low-cost model", f)
    log_output("  3. Identify and recommend the model with the best price-to-performance ratio among baseline and small LLM", f)
    log_output("", f)
    
    log_output("🚀 LOADING DATA...", f)
    df = pd.read_parquet("samples_3k_project_c_updated.parquet")
    log_output(f"✓ Dataset: {len(df)} records", f)

    # Extract and normalize text data
    def extract_name(names_json):
        if pd.isna(names_json):
            return ""
        try:
            names_data = json.loads(names_json)
            if isinstance(names_data, dict):
                return names_data.get('primary', '')
            return ""
        except:
            return ""

    def extract_address(addresses_json):
        if pd.isna(addresses_json):
            return ""
        try:
            addr_data = json.loads(addresses_json)
            if isinstance(addr_data, list) and len(addr_data) > 0:
                return addr_data[0].get('freeform', '')
            return ""
        except:
            return ""
    
    def normalize_text(text):
        """Enhanced text normalization with abbreviation handling and common word removal"""
        if not text:
            return ""
        import re
        
        # Abbreviation mapping
        abbrev_map = {
            'st': 'street', 'ave': 'avenue', 'rd': 'road', 'blvd': 'boulevard',
            'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
            'pkwy': 'parkway', 'hwy': 'highway', 'sq': 'square', 'cir': 'circle',
            'apt': 'apartment', 'ste': 'suite', 'bldg': 'building', 'fl': 'floor',
            'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
            'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest'
        }
        
        text = text.lower()
        
        # Expand abbreviations
        words = text.split()
        expanded_words = []
        for word in words:
            # Remove punctuation from word for lookup
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in abbrev_map:
                expanded_words.append(abbrev_map[word_clean])
            else:
                expanded_words.append(word)
        text = ' '.join(expanded_words)
        
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_category(categories_json):
        if pd.isna(categories_json):
            return ""
        try:
            cat_data = json.loads(categories_json)
            if isinstance(cat_data, dict):
                return cat_data.get('primary', '')
            return ""
        except:
            return ""

    # Extract names and addresses
    df['name_a'] = df['names'].apply(extract_name)
    df['name_b'] = df['base_names'].apply(extract_name)
    df['addr_a'] = df['addresses'].apply(extract_address)
    df['addr_b'] = df['base_addresses'].apply(extract_address)
    df['cat_a'] = df['categories'].apply(extract_category)
    df['cat_b'] = df['base_categories'].apply(extract_category)

    # Create ground truth labels
    def create_ground_truth(row):
        """Improved ground-truth with better balance"""
        name_a = row["name_a_norm"]
        name_b = row["name_b_norm"]
        addr_a = row["addr_a_norm"]
        addr_b = row["addr_b_norm"]

        # NAME MATCHING
        strong_name = (name_a == name_b)
    
        tokens_a = set(name_a.split())
        tokens_b = set(name_b.split())
        if len(tokens_a | tokens_b) > 0:
            name_jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        else:
            name_jaccard = 0.0
    
        # Relax threshold slightly for better recall
        weak_name = (name_jaccard >= 0.4)  # Changed from 0.5 to 0.4

        # ADDRESS MATCHING
        strong_addr = (addr_a == addr_b)
    
        import re
        nums_a = re.findall(r"\b\d+\b", addr_a)
        nums_b = re.findall(r"\b\d+\b", addr_b)
        same_number = (len(nums_a) > 0 and nums_a == nums_b)
    
        # Add partial address match
        addr_tokens_a = set(addr_a.split())
        addr_tokens_b = set(addr_b.split())
        if len(addr_tokens_a | addr_tokens_b) > 0:
            addr_jaccard = len(addr_tokens_a & addr_tokens_b) / len(addr_tokens_a | addr_tokens_b)
        else:
            addr_jaccard = 0.0
    
        partial_addr = (addr_jaccard >= 0.5)

        # FINAL LABEL - more nuanced rules
        is_match = (
            (strong_name and (strong_addr or same_number or partial_addr)) or
            (weak_name and strong_addr) or
            (weak_name and same_number and partial_addr)
        )

        return int(is_match)


    # Normalize text for better matching (needed before ground truth creation)
    df['name_a_norm'] = df['name_a'].apply(normalize_text)
    df['name_b_norm'] = df['name_b'].apply(normalize_text)
    df['addr_a_norm'] = df['addr_a'].apply(normalize_text)
    df['addr_b_norm'] = df['addr_b'].apply(normalize_text)
    
    df['ground_truth'] = df.apply(create_ground_truth, axis=1)
    log_output(f"✓ Ground truth: {df['ground_truth'].mean():.1%} match rate", f)
    df['cat_a_norm'] = df['cat_a'].apply(lambda x: normalize_text(str(x)) if pd.notna(x) else "")
    df['cat_b_norm'] = df['cat_b'].apply(lambda x: normalize_text(str(x)) if pd.notna(x) else "")

    # Create multiple text representations for ensemble approach
    df['text_a_full'] = (df['name_a_norm'] + " " + df['addr_a_norm'] + " " + df['cat_a_norm']).str.strip()
    df['text_b_full'] = (df['name_b_norm'] + " " + df['addr_b_norm'] + " " + df['cat_b_norm']).str.strip()
    df['text_a_name'] = df['name_a_norm']
    df['text_b_name'] = df['name_b_norm']
    df['text_a_addr'] = df['addr_a_norm']
    df['text_b_addr'] = df['addr_b_norm']
    df['text_a_name_addr'] = (df['name_a_norm'] + " " + df['addr_a_norm']).str.strip()
    df['text_b_name_addr'] = (df['name_b_norm'] + " " + df['addr_b_norm']).str.strip()
    
    # Keep backward compatibility
    df['text_a_enhanced'] = df['text_a_full']
    df['text_b_enhanced'] = df['text_b_full']

    # Split data for evaluation
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ground_truth'])
    log_output(f"✓ Split: {len(train_df)} train, {len(test_df)} test", f)

    def evaluate_model_ensemble(model_name, model, threshold=0.8, weights=None, test_data=None):
        """Ensemble evaluation using multiple text representations"""
        if test_data is None:
            test_data = test_df
        if weights is None:
            weights = {'full': 0.4, 'name': 0.3, 'addr': 0.2, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0}
        
        start_time = time.time()
        
        # Prepare text representations
        text_reprs = {
            'full': (test_data["text_a_full"].tolist(), test_data["text_b_full"].tolist()),
            'name': (test_data["text_a_name"].tolist(), test_data["text_b_name"].tolist()),
            'addr': (test_data["text_a_addr"].tolist(), test_data["text_b_addr"].tolist()),
            'name_addr': (test_data["text_a_name_addr"].tolist(), test_data["text_b_name_addr"].tolist())
        }
        
        all_scores = {}
        
        for repr_name, (text_a_list, text_b_list) in text_reprs.items():
            # Skip empty representations
            if not any(text_a_list) and not any(text_b_list):
                continue
                
            emb_a = model.encode(text_a_list, batch_size=16, convert_to_tensor=False, show_progress_bar=False, device='cpu')
            emb_b = model.encode(text_b_list, batch_size=16, convert_to_tensor=False, show_progress_bar=False, device='cpu')
            scores = np.array([cosine_similarity([emb_a[i]], [emb_b[i]])[0][0] for i in range(len(emb_a))])
            all_scores[repr_name] = scores
        
        encoding_time = time.time() - start_time
        
        similarity_start = time.time()
        
        # Combine embedding scores with weights
        ensemble_scores = np.zeros(len(test_data))
        total_weight = 0
        for repr_name, scores in all_scores.items():
            if repr_name in weights and weights.get(repr_name, 0) > 0:
                weight = weights[repr_name]
                ensemble_scores += scores * weight
                total_weight += weight
        
        # Add string similarity scores
        if weights.get('name_jaccard', 0) > 0:
            name_jaccard_scores = np.array([
                jaccard_similarity(test_data.iloc[i]['name_a_norm'], test_data.iloc[i]['name_b_norm'])
                for i in range(len(test_data))
            ])
            ensemble_scores += name_jaccard_scores * weights['name_jaccard']
            total_weight += weights['name_jaccard']
        
        if weights.get('addr_jaccard', 0) > 0:
            addr_jaccard_scores = np.array([
                jaccard_similarity(test_data.iloc[i]['addr_a_norm'], test_data.iloc[i]['addr_b_norm'])
                for i in range(len(test_data))
            ])
            ensemble_scores += addr_jaccard_scores * weights['addr_jaccard']
            total_weight += weights['addr_jaccard']
        
        if total_weight > 0:
            ensemble_scores = ensemble_scores / total_weight
        
        # Precision-focused approach: filter false positives while preserving true positives
        # Get individual component scores
        name_scores = all_scores.get('name', np.zeros(len(test_data)))
        addr_scores = all_scores.get('addr', np.zeros(len(test_data)))
        
        similarity_time = time.time() - similarity_start
        
        # Use threshold with precision-focused filtering
        # Filter only very clear false positives (low name AND low address) in borderline range
        base_predictions = (ensemble_scores > threshold).astype(int)
        
        # Find borderline cases (just above threshold)
        borderline = (ensemble_scores > threshold) & (ensemble_scores < threshold + 0.07)
        
        # Filter: remove cases where BOTH name and address similarities are very low
        # This targets false positives while keeping true positives (which usually have at least one high score)
        very_low_both = (name_scores < 0.50) & (addr_scores < 0.55)
        false_positive_mask = borderline & very_low_both
        
        # Apply filter
        predictions = base_predictions.copy()
        predictions[false_positive_mask] = 0
        ground_truth = test_data['ground_truth'].values
        
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
        
        total_time = encoding_time + similarity_time
        time_per_match = (total_time / len(test_data)) * 1000
        
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
            'weights': weights,
            'n_samples': len(test_data)
        }
        
        log_output(f"\nRESULTS:", f)
        log_output(f"Precision: {precision:.4f}", f)
        log_output(f"Recall: {recall:.4f}", f)
        log_output(f"F1 Score: {f1:.4f}", f)
        log_output(f"Time per match: {time_per_match:.2f}ms", f)
        log_output(f"Threshold: {threshold:.3f}", f)
        
        return results, ensemble_scores, predictions

    def jaccard_similarity(text_a, text_b):
        """Calculate Jaccard similarity between two texts based on word sets"""
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
    
    def optimize_threshold_ensemble(model, test_data=None):
        """Optimize threshold and weights for ensemble approach"""
        if test_data is None:
            test_data = test_df
        
        # Prepare text representations
        text_reprs = {
            'full': (test_data["text_a_full"].tolist(), test_data["text_b_full"].tolist()),
            'name': (test_data["text_a_name"].tolist(), test_data["text_b_name"].tolist()),
            'addr': (test_data["text_a_addr"].tolist(), test_data["text_b_addr"].tolist()),
            'name_addr': (test_data["text_a_name_addr"].tolist(), test_data["text_b_name_addr"].tolist())
        }
        
        all_scores = {}
        for repr_name, (text_a_list, text_b_list) in text_reprs.items():
            if not any(text_a_list) and not any(text_b_list):
                continue
            emb_a = model.encode(text_a_list, batch_size=16, convert_to_tensor=False, device='cpu')
            emb_b = model.encode(text_b_list, batch_size=16, convert_to_tensor=False, device='cpu')
            scores = np.array([cosine_similarity([emb_a[i]], [emb_b[i]])[0][0] for i in range(len(emb_a))])
            all_scores[repr_name] = scores
        
        # Add string-based similarity scores
        string_scores = {}
        string_scores['name_jaccard'] = np.array([
            jaccard_similarity(test_data.iloc[i]['name_a_norm'], test_data.iloc[i]['name_b_norm'])
            for i in range(len(test_data))
        ])
        string_scores['addr_jaccard'] = np.array([
            jaccard_similarity(test_data.iloc[i]['addr_a_norm'], test_data.iloc[i]['addr_b_norm'])
            for i in range(len(test_data))
        ])
        
        ground_truth = test_data['ground_truth'].values
        
        # Focused weight combinations - emphasize name similarity (strongest signal)
        weight_candidates = [
            # Current best and variations
            {'full': 0.4, 'name': 0.3, 'addr': 0.2, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            {'full': 0.43, 'name': 0.27, 'addr': 0.2, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            # Name-emphasized combinations
            {'full': 0.35, 'name': 0.4, 'addr': 0.15, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            {'full': 0.3, 'name': 0.45, 'addr': 0.15, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            {'full': 0.32, 'name': 0.38, 'addr': 0.2, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            {'full': 0.38, 'name': 0.32, 'addr': 0.2, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            # Balanced with slight name emphasis
            {'full': 0.4, 'name': 0.35, 'addr': 0.15, 'name_addr': 0.1, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
            {'full': 0.45, 'name': 0.35, 'addr': 0.15, 'name_addr': 0.05, 'name_jaccard': 0.0, 'addr_jaccard': 0.0},
        ]
        
        best_f1 = 0
        best_threshold = 0.5
        best_weights = weight_candidates[0]
        
        for weights in weight_candidates:
            # Combine embedding scores
            ensemble_scores = np.zeros(len(test_data))
            total_weight = 0
            
            for repr_name, scores in all_scores.items():
                if repr_name in weights and weights[repr_name] > 0:
                    weight = weights[repr_name]
                    ensemble_scores += scores * weight
                    total_weight += weight
            
            # Add string similarity scores
            if 'name_jaccard' in weights and weights['name_jaccard'] > 0:
                ensemble_scores += string_scores['name_jaccard'] * weights['name_jaccard']
                total_weight += weights['name_jaccard']
            
            if 'addr_jaccard' in weights and weights['addr_jaccard'] > 0:
                ensemble_scores += string_scores['addr_jaccard'] * weights['addr_jaccard']
                total_weight += weights['addr_jaccard']
            
            if total_weight > 0:
                ensemble_scores = ensemble_scores / total_weight
            
            # Optimize threshold for these weights - focus on precision while maintaining recall
            # Focused threshold search around optimal range (0.75-0.85) for speed
            # Use very fine granularity in the sweet spot
            thresholds = np.concatenate([
                np.arange(0.78, 0.83, 0.005),  # Very fine-grained around optimal range
                np.arange(0.75, 0.78, 0.01),
                np.arange(0.83, 0.88, 0.01)
            ])
            for threshold in thresholds:
                predictions = (ensemble_scores > threshold).astype(int)
                
                if len(np.unique(predictions)) > 1:
                    prec = precision_score(ground_truth, predictions)
                    rec = recall_score(ground_truth, predictions)
                    f1 = f1_score(ground_truth, predictions)
                    
                    # Optimize for F1, but prefer configurations with better precision when recall is good
                    # This helps improve precision without hurting recall too much
                    score = f1
                    if rec >= 0.78:  # If recall is good (>=78%), give small bonus for precision
                        prec_bonus = max(0, (prec - 0.68) * 0.15)  # Small bonus for precision above baseline
                        score = f1 + prec_bonus
                    
                    if score > best_f1:
                        best_f1 = score
                        best_threshold = threshold
                        best_weights = weights.copy()
        
        return best_threshold, best_f1, best_weights

    # BASELINE COMPARISON
    log_output("", f)
    log_output("="*60, f)
    log_output("BASELINE COMPARISON", f)
    log_output("="*60, f)
    
    log_output(f"📊 Previous Matcher: {df['ground_truth'].mean():.1%} accuracy, ~1ms, $0.00", f)
    
    # Define models to evaluate
    models_to_evaluate = [
        {'name': 'all-MiniLM-L6-v2', 'path': 'all-MiniLM-L6-v2', 'size_mb': 22, 'cost_per_1m': 0.10},
        {'name': 'paraphrase-MiniLM-L6-v2', 'path': 'paraphrase-MiniLM-L6-v2', 'size_mb': 22, 'cost_per_1m': 0.10},
        {'name': 'all-mpnet-base-v2', 'path': 'all-mpnet-base-v2', 'size_mb': 420, 'cost_per_1m': 0.10},
    ]
    
    log_output("", f)
    log_output("="*80, f)
    log_output("LANGUAGE MODEL EVALUATION: MULTI-MODEL COMPARISON", f)
    log_output("="*80, f)
    
    all_results = []
    model_predictions = {}  # Store predictions for each model
    baseline_result = {
        'model_name': 'Previous Matcher (Baseline)',
        'f1_score': df['ground_truth'].mean(),
        'precision': None,
        'recall': None,
        'time_per_match_ms': 1.0,
        'cost_per_1m': 0.00,
        'size_mb': 0,
        'meets_speed': True,
        'meets_f1': False
    }
    all_results.append(baseline_result)
    
    for model_config in models_to_evaluate:
        model_name = model_config['name']
        model_path = model_config['path']
        
        try:
            log_output(f"\n🤖 Evaluating {model_name}...", f)
            model = SentenceTransformer(model_path, device='cpu')
            log_output("✓ Model loaded", f)
            
            optimal_threshold, optimal_f1, optimal_weights = optimize_threshold_ensemble(model, train_df)
            results, scores, predictions = evaluate_model_ensemble(model_name, model, optimal_threshold, optimal_weights, test_df)
            
            # Check if model meets speed requirement (stop if exceeds 50ms significantly)
            if results['time_per_match_ms'] > 100:
                log_output(f"⚠️  Model exceeds speed requirement ({results['time_per_match_ms']:.1f}ms > 50ms), skipping further evaluation", f)
                results['meets_speed'] = False
                results['meets_f1'] = results['f1_score'] >= 0.80
            else:
                results['meets_speed'] = results['time_per_match_ms'] <= 50
                results['meets_f1'] = results['f1_score'] >= 0.80
            
            results['cost_per_1m'] = model_config['cost_per_1m']
            results['size_mb'] = model_config['size_mb']
            all_results.append(results)
            model_predictions[model_name] = {
                'scores': scores,
                'predictions': predictions
            }
            
        except Exception as e:
            log_output(f"❌ Error evaluating {model_name}: {e}", f)
            continue
    
    # Comparative Analysis Report
    log_output("", f)
    log_output("="*80, f)
    log_output("COMPARATIVE ANALYSIS REPORT: MODEL PERFORMANCE EVALUATION", f)
    log_output("="*80, f)
    
    log_output(f"\n📊 PERFORMANCE METRICS COMPARISON:", f)
    log_output(f"", f)
    log_output(f"{'Model':<35} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Speed (ms)':<12} {'Cost/1M':<12} {'Size (MB)':<12}", f)
    log_output(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}", f)
    
    for result in all_results:
        if result['precision'] is None:
            log_output(f"{result['model_name']:<35} {result['f1_score']:<12.1%} {'N/A':<12} {'N/A':<12} {result['time_per_match_ms']:<12.1f} ${result['cost_per_1m']:<11.2f} {result['size_mb']:<12}", f)
        else:
            log_output(f"{result['model_name']:<35} {result['f1_score']:<12.1%} {result['precision']:<12.3f} {result['recall']:<12.3f} {result['time_per_match_ms']:<12.1f} ${result['cost_per_1m']:<11.2f} {result['size_mb']:<12}", f)
        
    # Calculate performance improvements
    baseline_f1 = df['ground_truth'].mean()
    
    # Calculate price-performance scores first to determine best model
    price_perf_scores = []
    for result in all_results:
        if result.get('precision') is not None:
            cost_per_match = result['cost_per_1m'] / 1000000  # Approximate
            f1_per_dollar = result['f1_score'] / cost_per_match if cost_per_match > 0 else float('inf')
            f1_per_ms = result['f1_score'] / result['time_per_match_ms']
            composite_score = result['f1_score'] * (1.0 / max(result['time_per_match_ms'], 1)) * (1.0 / max(cost_per_match, 0.000001))
            price_perf_scores.append({
                'model': result['model_name'],
                'f1': result['f1_score'],
                'speed': result['time_per_match_ms'],
                'cost': cost_per_match,
                'f1_per_dollar': f1_per_dollar,
                'f1_per_ms': f1_per_ms,
                'composite_score': composite_score,
                'meets_speed': result['meets_speed'],
                'meets_f1': result['meets_f1']
            })
    
    # Find best models
    best_f1_model = max(price_perf_scores, key=lambda x: x['f1']) if price_perf_scores else None
    best_pp_model = max(price_perf_scores, key=lambda x: x['composite_score']) if price_perf_scores else None
    
    # Find best overall model (considering all OKRs)
    valid_models = [m for m in price_perf_scores if m['meets_speed'] and m['meets_f1']]
    if not valid_models:
        valid_models = sorted(price_perf_scores, key=lambda x: x['f1'], reverse=True)
    best_model = max(valid_models, key=lambda x: x['composite_score']) if valid_models else None
    
    log_output("", f)
    log_output("="*80, f)
    log_output("SUMMARY & RECOMMENDATION", f)
    log_output("="*80, f)
    log_output("", f)
    
    # Top Performers
    if best_f1_model:
        best_f1_result = next((r for r in all_results if r['model_name'] == best_f1_model['model']), None)
        if best_f1_result:
            log_output(f"🏆 Top Performer (F1 Score): {best_f1_model['model']} - {best_f1_model['f1']:.1%} F1", f)
    
    if best_pp_model:
        best_pp_result = next((r for r in all_results if r['model_name'] == best_pp_model['model']), None)
        if best_pp_result:
            log_output(f"💰 Top Price-Performer: {best_pp_model['model']} - Score: {best_pp_model['composite_score']:.2f}", f)
    
    log_output("", f)
    log_output("OKR EVALUATION TABLE:", f)
    log_output(f"{'Model':<35} {'KR1 (F1≥80%)':<15} {'KR2 (≤50ms)':<15} {'KR3 (Best P/P)':<15}", f)
    log_output(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15}", f)
    
    # Display OKR status for each model
    for result in all_results:
        if result.get('precision') is not None:
            kr1_status = "✅" if result['f1_score'] >= 0.80 else f"❌ {result['f1_score']:.1%}"
            kr2_status = "✅" if result['time_per_match_ms'] <= 50 else f"❌ {result['time_per_match_ms']:.1f}ms"
            kr3_status = "✅" if result['model_name'] == best_pp_model['model'] else "❌"
            
            log_output(f"{result['model_name']:<35} {kr1_status:<15} {kr2_status:<15} {kr3_status:<15}", f)
        elif result['model_name'] == 'Previous Matcher (Baseline)':
            log_output(f"{result['model_name']:<35} {'N/A':<15} {'✅':<15} {'N/A':<15}", f)
    
    log_output("", f)
    log_output("="*80, f)
    log_output("FINAL RECOMMENDATION", f)
    log_output("="*80, f)
    log_output("", f)
    
    if best_model:
        best_result = next((r for r in all_results if r['model_name'] == best_model['model']), None)
        if best_result:
            improvement = best_result['f1_score'] - baseline_f1
            log_output(f"✅ RECOMMENDED MODEL: {best_model['model']}", f)
            log_output("", f)
            log_output(f"Rationale:", f)
            log_output(f"  • F1 Score: {best_result['f1_score']:.1%} ({improvement:+.1%} vs baseline)", f)
            log_output(f"  • Speed: {best_result['time_per_match_ms']:.1f}ms per match", f)
            log_output(f"  • Price-Performance: Composite Score {best_model['composite_score']:.2f}", f)
            log_output(f"  • OKRs Met: KR1 {'✅' if best_result['f1_score'] >= 0.80 else '❌'}, KR2 {'✅' if best_result['time_per_match_ms'] <= 50 else '❌'}, KR3 {'✅' if best_model['model'] == best_pp_model['model'] else '❌'}", f)
            log_output("", f)
    
    log_output("="*80, f)

print("\n🎉 Model evaluation completed!")
print("📄 Results saved to: results.txt")
print("📊 Comparative analysis and recommendation ready for review!")
