"""
Mining Big Data - Assignment 3
Task A: Pattern Mining

This script demonstrates pattern mining techniques using both the Apriori and FP-Growth 
algorithms on grocery store transaction data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our pattern mining module
from src.task_a.pattern_mining import PatternMiner
from utils.data_utils import preprocess_data, plot_purchase_frequency, plot_purchase_trends

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*80)
    print("Mining Big Data - Assignment 3 - Task A: Pattern Mining")
    print("="*80)
    
    # 1. Load the dataset
    print("\n1. Loading dataset...")
    train_path = os.path.join('dataset', 'train.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: Dataset file not found at {train_path}")
        sys.exit(1)
    
    try:
        train_data = pd.read_csv(train_path)
        print(f"Successfully loaded training data: {train_data.shape[0]} records")
        print(f"Number of unique users: {train_data['User_id'].nunique()}")
        print(f"Number of unique items: {train_data['itemDescription'].nunique()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # 2. Display dataset information
    print("\n2. Dataset information:")
    print(train_data.head())
    print("\nData types:")
    print(train_data.dtypes)
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    preprocessed_data = preprocess_data(train_data)
    print("Preprocessing complete.")
    
    # 4. Generate exploratory visualizations
    print("\n4. Generating exploratory visualizations...")
    purchase_freq_path = plot_purchase_frequency(preprocessed_data, top_n=15)
    purchase_trends_path = plot_purchase_trends(preprocessed_data, interval='M')
    print(f"Purchase frequency visualization saved to: {purchase_freq_path}")
    print(f"Purchase trends visualization saved to: {purchase_trends_path}")
    
    # 5. Pattern Mining with Apriori - IMPROVED PARAMETERS
    print("\n5. Running Apriori algorithm with improved parameters...")
    apriori_miner = PatternMiner(train_data)
    apriori_start = time.time()
    # Lower minimum support and confidence thresholds to discover more patterns
    apriori_miner.run(algorithm='apriori', min_support=0.005, min_confidence=0.2, min_lift=1.0)
    apriori_time = time.time() - apriori_start
    
    # 6. Pattern Mining with FP-Growth - IMPROVED PARAMETERS
    print("\n6. Running FP-Growth algorithm with improved parameters...")
    fpgrowth_miner = PatternMiner(train_data)
    fpgrowth_start = time.time()
    # Use same parameters for fair comparison
    fpgrowth_miner.run(algorithm='fpgrowth', min_support=0.005, min_confidence=0.2, min_lift=1.0)
    fpgrowth_time = time.time() - fpgrowth_start
    
    # 7. Compare algorithms
    print("\n7. Algorithm comparison:")
    print(f"Apriori execution time: {apriori_time:.2f} seconds")
    print(f"FP-Growth execution time: {fpgrowth_time:.2f} seconds")
    
    if fpgrowth_time > 0:
        speedup = apriori_time / fpgrowth_time
        print(f"FP-Growth is {speedup:.2f}x faster than Apriori")
    
    # 8. Display frequent itemsets with improved display for length > 1
    print("\n8. Top frequent itemsets (by support):")
    if apriori_miner.frequent_itemsets is not None and len(apriori_miner.frequent_itemsets) > 0:
        display_itemsets = apriori_miner.frequent_itemsets.copy()
        display_itemsets['itemsets_str'] = display_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        display_itemsets['length'] = display_itemsets['itemsets'].apply(len)
        
        # Show top itemsets by support
        print("\n8.1 Top itemsets by support (any length):")
        top_itemsets = display_itemsets.sort_values('support', ascending=False).head(10)
        for i, row in top_itemsets.iterrows():
            print(f"Support: {row['support']:.4f}, Length: {row['length']}, Items: {row['itemsets_str']}")
        
        # Show top itemsets of length > 1
        print("\n8.2 Top itemsets with multiple items:")
        multi_itemsets = display_itemsets[display_itemsets['length'] > 1].sort_values('support', ascending=False).head(10)
        if len(multi_itemsets) > 0:
            for i, row in multi_itemsets.iterrows():
                print(f"Support: {row['support']:.4f}, Length: {row['length']}, Items: {row['itemsets_str']}")
        else:
            print("No itemsets with multiple items found. Try further reducing the minimum support threshold.")
    else:
        print("No frequent itemsets found. Try lowering the minimum support threshold.")
    
    # 9. Display association rules
    print("\n9. Top association rules (by lift):")
    if apriori_miner.association_rules is not None and len(apriori_miner.association_rules) > 0:
        display_rules = apriori_miner.association_rules.copy()
        display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents_str'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        top_rules = display_rules.sort_values('lift', ascending=False).head(10)
        for i, row in top_rules.iterrows():
            print(f"Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}, Support: {row['support']:.4f}")
            print(f"Rule: {row['antecedents_str']} -> {row['consequents_str']}")
            print("-" * 50)
    else:
        print("No association rules found. Try adjusting the minimum confidence threshold.")
    
    # 10. Visualize results
    print("\n10. Generating visualizations...")
    apriori_miner.visualize_results()
    
    # 11. User-specific patterns
    print("\n11. User-specific pattern mining:")
    # Select a random user ID
    random_user_id = np.random.choice(train_data['User_id'].unique())
    print(f"Selected user ID: {random_user_id}")
    
    # Get user's purchase history
    user_data = train_data[train_data['User_id'] == random_user_id]
    print(f"User has made {len(user_data)} purchases of {user_data['itemDescription'].nunique()} unique items")
    
    # Display user's most purchased items
    user_items = user_data['itemDescription'].value_counts().head(5)
    print("\nUser's top purchased items:")
    for item, count in user_items.items():
        print(f"  - {item}: {count} times")
    
    # Get patterns relevant to the user
    user_patterns = apriori_miner.get_patterns_for_user(random_user_id, top_n=5)
    
    if len(user_patterns) > 0:
        # Convert frozensets to strings for better display
        user_patterns['itemsets_str'] = user_patterns['itemsets'].apply(lambda x: ', '.join(list(x)))
        user_patterns['length'] = user_patterns['itemsets'].apply(len)
        
        # Display relevant patterns
        print(f"\nTop patterns relevant to user {random_user_id}:")
        for i, row in user_patterns.iterrows():
            print(f"Score: {row['score']:.4f}, Support: {row['support']:.4f}, Length: {row['length']}, Items: {row['itemsets_str']}")
    else:
        print("\nNo relevant patterns found for this user.")
    
    # 12. Generate additional visualizations for improved insights
    print("\n12. Generating additional visualizations for improved insights...")
    
    # 12.1 Create plot for itemset length distribution
    if apriori_miner.frequent_itemsets is not None and len(apriori_miner.frequent_itemsets) > 0:
        plt.figure(figsize=(10, 6))
        length_dist = apriori_miner.frequent_itemsets['length'].value_counts().sort_index()
        plt.bar(length_dist.index, length_dist.values)
        plt.title('Improved Itemset Length Distribution')
        plt.xlabel('Itemset Length')
        plt.ylabel('Count')
        plt.xticks(length_dist.index)
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/improved_length_distribution.png')
        plt.close()
        print("Created improved length distribution visualization")
    
    # 13. Export results
    print("\n13. Exporting results...")
    apriori_miner.export_results(output_dir='results')
    fpgrowth_miner.export_results(output_dir='results')
    
    print("\nTask A complete! Results are available in the 'results' and 'visualizations' directories.")

if __name__ == "__main__":
    main() 