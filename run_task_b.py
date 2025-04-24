"""
Mining Big Data - Assignment 3
Task B: Collaborative Filtering

This script demonstrates collaborative filtering techniques for building a recommendation system.
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

# Import our collaborative filtering module
from src.task_b.collaborative_filtering import CollaborativeFilter
from utils.data_utils import preprocess_data

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*80)
    print("Mining Big Data - Assignment 3 - Task B: Collaborative Filtering")
    print("="*80)
    
    # 1. Load the dataset
    print("\n1. Loading dataset...")
    train_path = os.path.join('dataset', 'train.csv')
    test_path = os.path.join('dataset', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Dataset files not found")
        sys.exit(1)
    
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"Successfully loaded training data: {train_data.shape[0]} records")
        print(f"Successfully loaded test data: {test_data.shape[0]} records")
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
    preprocessed_test_data = preprocess_data(test_data)
    print("Preprocessing complete.")
    
    # 4. Initialize Collaborative Filtering
    print("\n4. Initializing Collaborative Filtering...")
    cf = CollaborativeFilter(train_data)
    
    # 5. Prepare user-item matrix
    print("\n5. Preparing user-item matrix...")
    cf.prepare_data()
    
    # 6. Compute similarity matrices
    print("\n6. Computing similarity matrices...")
    print("\n6.1 Computing user similarity matrix...")
    cf.compute_similarity(mode='user')
    
    print("\n6.2 Computing item similarity matrix...")
    cf.compute_similarity(mode='item')
    
    # 7. Train SVD model
    print("\n7. Training SVD model...")
    cf.train_svd_model(n_components=50)
    
    # 8. Generate recommendations using different methods
    print("\n8. Generating recommendations using different methods...")
    
    # Randomly select a user ID from the data
    user_ids = train_data['User_id'].unique()
    selected_user_id = np.random.choice(user_ids)
    
    print(f"\n8.1 User-based collaborative filtering for user {selected_user_id}:")
    start_time = time.time()
    user_based_recommendations = cf.generate_recommendations(
        user_id=selected_user_id, 
        method='user-based',
        n_recommendations=5
    )
    user_based_time = time.time() - start_time
    
    print(f"\n8.2 Item-based collaborative filtering for user {selected_user_id}:")
    start_time = time.time()
    item_based_recommendations = cf.generate_recommendations(
        user_id=selected_user_id, 
        method='item-based',
        n_recommendations=5
    )
    item_based_time = time.time() - start_time
    
    print(f"\n8.3 SVD-based collaborative filtering for user {selected_user_id}:")
    start_time = time.time()
    svd_recommendations = cf.generate_recommendations(
        user_id=selected_user_id, 
        method='svd',
        n_recommendations=5
    )
    svd_time = time.time() - start_time
    
    # 9. Compare recommendation methods execution time
    print("\n9. Recommendation methods execution time comparison:")
    print(f"User-based: {user_based_time:.4f} seconds")
    print(f"Item-based: {item_based_time:.4f} seconds")
    print(f"SVD-based: {svd_time:.4f} seconds")
    
    # 10. Evaluate recommendation quality
    print("\n10. Evaluating recommendation quality...")
    
    # Initialize metrics dictionaries with default values
    user_based_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    item_based_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    svd_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    
    try:
        print("\n10.1 User-based recommendation evaluation:")
        user_based_metrics = cf.evaluate_recommendations(
            test_data=test_data,
            n_users=50,
            method='user-based'
        )
    except Exception as e:
        print(f"Error evaluating user-based recommendations: {e}")
    
    try:
        print("\n10.2 Item-based recommendation evaluation:")
        item_based_metrics = cf.evaluate_recommendations(
            test_data=test_data,
            n_users=50,
            method='item-based'
        )
    except Exception as e:
        print(f"Error evaluating item-based recommendations: {e}")
    
    try:
        print("\n10.3 SVD-based recommendation evaluation:")
        svd_metrics = cf.evaluate_recommendations(
            test_data=test_data,
            n_users=50,
            method='svd'
        )
    except Exception as e:
        print(f"Error evaluating SVD-based recommendations: {e}")
    
    # 11. Compare all recommendation methods
    print("\n11. Comprehensive method comparison:")
    try:
        comparison_results = cf.compare_methods(
            user_id=selected_user_id,
            n_recommendations=5,
            test_data=test_data
        )
    except Exception as e:
        print(f"Error comparing recommendation methods: {e}")
        comparison_results = {
            'user_id': selected_user_id,
            'recommendations': {},
            'execution_times': {}
        }
    
    # 12. Generate visualizations
    print("\n12. Generating visualizations...")
    cf.visualize_results()
    
    # 13. Visualize recommendation quality comparison
    print("\n13. Visualizing recommendation quality comparison...")
    
    # Create bar plot for metrics comparison
    methods = ['User-based', 'Item-based', 'SVD-based']
    hit_rates = [user_based_metrics.get('hit_rate', 0), item_based_metrics.get('hit_rate', 0), svd_metrics.get('hit_rate', 0)]
    precisions = [user_based_metrics.get('average_precision', 0), item_based_metrics.get('average_precision', 0), svd_metrics.get('average_precision', 0)]
    coverage = [user_based_metrics.get('coverage', 0), item_based_metrics.get('coverage', 0), svd_metrics.get('coverage', 0)]
    diversity = [user_based_metrics.get('diversity', 0), item_based_metrics.get('diversity', 0), svd_metrics.get('diversity', 0)]
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(methods, hit_rates)
    plt.title('Hit Rate Comparison')
    plt.ylabel('Hit Rate')
    
    plt.subplot(2, 2, 2)
    plt.bar(methods, precisions)
    plt.title('Precision Comparison')
    plt.ylabel('Precision')
    
    plt.subplot(2, 2, 3)
    plt.bar(methods, coverage)
    plt.title('Coverage Comparison')
    plt.ylabel('Coverage')
    
    plt.subplot(2, 2, 4)
    plt.bar(methods, diversity)
    plt.title('Diversity Comparison')
    plt.ylabel('Diversity')
    
    plt.tight_layout()
    plt.savefig('visualizations/recommendation_metrics_comparison.png')
    plt.close()
    
    # 14. Export results
    print("\n14. Exporting results...")
    
    # Export metrics comparison to CSV
    metrics_comparison = pd.DataFrame({
        'Method': methods,
        'Hit Rate': hit_rates,
        'Precision': precisions,
        'Coverage': coverage,
        'Diversity': diversity
    })
    
    metrics_comparison.to_csv('results/recommendation_metrics_comparison.csv', index=False)
    
    # Create summary file
    with open('results/collaborative_filtering_summary.txt', 'w') as f:
        f.write("==========================================================\n")
        f.write("COLLABORATIVE FILTERING RECOMMENDATION SYSTEM SUMMARY\n")
        f.write("==========================================================\n\n")
        
        f.write(f"User ID: {selected_user_id}\n\n")
        
        f.write("1. User-Based Collaborative Filtering Recommendations:\n")
        for i, (item, score) in enumerate(user_based_recommendations.items()):
            f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
        f.write("\n")
        
        f.write("2. Item-Based Collaborative Filtering Recommendations:\n")
        for i, (item, score) in enumerate(item_based_recommendations.items()):
            f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
        f.write("\n")
        
        f.write("3. SVD-Based Collaborative Filtering Recommendations:\n")
        for i, (item, score) in enumerate(svd_recommendations.items()):
            f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
        f.write("\n")
        
        f.write("4. Performance Comparison:\n")
        f.write(f"   User-based execution time: {user_based_time:.4f} seconds\n")
        f.write(f"   Item-based execution time: {item_based_time:.4f} seconds\n")
        f.write(f"   SVD-based execution time: {svd_time:.4f} seconds\n\n")
        
        f.write("5. Quality Metrics Comparison:\n")
        f.write(f"   User-based: Hit Rate={user_based_metrics['hit_rate']:.4f}, Precision={user_based_metrics['average_precision']:.4f}, Coverage={user_based_metrics['coverage']:.4f}, Diversity={user_based_metrics['diversity']:.4f}\n")
        f.write(f"   Item-based: Hit Rate={item_based_metrics['hit_rate']:.4f}, Precision={item_based_metrics['average_precision']:.4f}, Coverage={item_based_metrics['coverage']:.4f}, Diversity={item_based_metrics['diversity']:.4f}\n")
        f.write(f"   SVD-based: Hit Rate={svd_metrics['hit_rate']:.4f}, Precision={svd_metrics['average_precision']:.4f}, Coverage={svd_metrics['coverage']:.4f}, Diversity={svd_metrics['diversity']:.4f}\n")
    
    print("\nTask B complete! Results are available in the 'results' and 'visualizations' directories.")
    print("Check the 'results/collaborative_filtering_summary.txt' file for a detailed summary.")

if __name__ == "__main__":
    main() 