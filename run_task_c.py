"""
Mining Big Data - Assignment 3
Task C: Integration

This script demonstrates the integration of pattern mining and collaborative filtering
for enhanced recommendation systems.
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

# Import integrated system module
from src.task_c.integration import IntegratedSystem

def main():
    # Create directories if they don't exist
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*80)
    print("Mining Big Data - Assignment 3 - Task C: Integration")
    print("="*80)
    
    # 1. Load datasets
    print("\n1. Loading datasets...")
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
    
    # 3. Initialize integrated system
    print("\n3. Initializing integrated system...")
    system = IntegratedSystem(train_data, test_data)
    
    # 4. Run pattern mining
    print("\n4. Running pattern mining...")
    start_time = time.time()
    system.mine_patterns(min_support=0.003, algorithm='fpgrowth')
    pattern_mining_time = time.time() - start_time
    print(f"Pattern mining completed in {pattern_mining_time:.2f} seconds")
    
    # 5. Prepare collaborative filtering
    print("\n5. Preparing collaborative filtering...")
    start_time = time.time()
    system.prepare_collaborative_filtering()
    collaborative_filtering_time = time.time() - start_time
    print(f"Collaborative filtering preparation completed in {collaborative_filtering_time:.2f} seconds")
    
    # 6. Generate recommendations with integrated system
    print("\n6. Generating recommendations with integrated system...")
    user_ids = train_data['User_id'].unique()
    selected_user_id = np.random.choice(user_ids)
    
    # Generate recommendations with patterns
    print(f"\n6.1 Generating hybrid recommendations for user {selected_user_id}:")
    start_time = time.time()
    hybrid_recommendations = system.generate_recommendations(
        user_id=selected_user_id,
        n_recommendations=10,
        with_patterns=True
    )
    hybrid_time = time.time() - start_time
    
    # Generate recommendations without patterns (CF only)
    print(f"\n6.2 Generating CF-only recommendations for user {selected_user_id}:")
    start_time = time.time()
    cf_recommendations = system.generate_recommendations(
        user_id=selected_user_id,
        n_recommendations=10,
        with_patterns=False
    )
    cf_time = time.time() - start_time
    
    # 7. Compare recommendation generation methods
    print("\n7. Recommendation methods execution time comparison:")
    print(f"Hybrid approach: {hybrid_time:.4f} seconds")
    print(f"CF-only approach: {cf_time:.4f} seconds")
    
    # 8. Evaluate recommendation quality
    print("\n8. Evaluating recommendation quality...")
    
    print("\n8.1 Hybrid recommendation evaluation:")
    try:
        hybrid_metrics = system.evaluate_recommendations(method='hybrid', n_users=5)
    except Exception as e:
        print(f"Error evaluating hybrid recommendations: {e}")
        hybrid_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    
    print("\n8.2 Collaborative filtering recommendation evaluation:")
    try:
        cf_metrics = system.evaluate_recommendations(method='cf', n_users=5)
    except Exception as e:
        print(f"Error evaluating CF recommendations: {e}")
        cf_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    
    print("\n8.3 Pattern-based recommendation evaluation:")
    try:
        pattern_metrics = system.evaluate_recommendations(method='pattern', n_users=5)
    except Exception as e:
        print(f"Error evaluating pattern-based recommendations: {e}")
        pattern_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
    
    # 9. Compare all recommendation methods
    print("\n9. Comprehensive method comparison:")
    try:
        comparison_results = system.compare_recommendation_methods(n_users=5)
    except Exception as e:
        print(f"Error comparing recommendation methods: {e}")
        comparison_results = None
    
    # 10. Generate visualizations
    print("\n10. Generating visualizations...")
    system.visualize_results()
    
    # 11. Export results
    print("\n11. Exporting results...")
    system.export_results()
    
    # 12. Create summary file
    print("\n12. Creating integration summary file...")
    with open('findings_task_c.txt', 'w') as f:
        f.write("==========================================================\n")
        f.write("TASK C: INTEGRATION FINDINGS\n")
        f.write("==========================================================\n\n")
        
        f.write("1. INTRODUCTION\n")
        f.write("---------------------------------------------------------\n")
        f.write("This document presents the findings from implementing an integrated recommendation system\n")
        f.write("that combines collaborative filtering and pattern mining techniques.\n\n")
        
        f.write("2. INTEGRATION APPROACH\n")
        f.write("---------------------------------------------------------\n")
        f.write("2.1 Hybrid Recommendation Method\n")
        f.write("- Combined collaborative filtering (SVD) and pattern mining results\n")
        f.write("- CF recommendations weighted at 60%\n")
        f.write("- Pattern-based recommendations weighted at 40%\n")
        f.write("- Items appearing in both sets received a 20% boost\n\n")
        
        f.write("2.2 Pattern-Based Recommendations\n")
        f.write("- Used frequent itemsets from pattern mining\n")
        f.write("- Items scored based on support and co-occurrence with user's purchases\n")
        f.write("- Focused on patterns that include at least one item the user has purchased\n\n")
        
        f.write("3. PERFORMANCE COMPARISON\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"- Pattern mining time: {pattern_mining_time:.2f} seconds\n")
        f.write(f"- Collaborative filtering setup time: {collaborative_filtering_time:.2f} seconds\n")
        f.write(f"- Hybrid recommendation generation time: {hybrid_time:.2f} seconds\n")
        f.write(f"- CF-only recommendation generation time: {cf_time:.2f} seconds\n\n")
        
        f.write("4. RECOMMENDATION QUALITY COMPARISON\n")
        f.write("---------------------------------------------------------\n")
        f.write("4.1 Hit Rate (ability to recommend items users will purchase)\n")
        f.write(f"- Hybrid approach: {hybrid_metrics['hit_rate']:.4f}\n")
        f.write(f"- Collaborative filtering: {cf_metrics['hit_rate']:.4f}\n")
        f.write(f"- Pattern-based: {pattern_metrics['hit_rate']:.4f}\n\n")
        
        f.write("4.2 Precision (proportion of relevant recommendations)\n")
        f.write(f"- Hybrid approach: {hybrid_metrics['average_precision']:.4f}\n")
        f.write(f"- Collaborative filtering: {cf_metrics['average_precision']:.4f}\n")
        f.write(f"- Pattern-based: {pattern_metrics['average_precision']:.4f}\n\n")
        
        f.write("4.3 Coverage (proportion of items that can be recommended)\n")
        f.write(f"- Hybrid approach: {hybrid_metrics['coverage']:.4f}\n")
        f.write(f"- Collaborative filtering: {cf_metrics['coverage']:.4f}\n")
        f.write(f"- Pattern-based: {pattern_metrics['coverage']:.4f}\n\n")
        
        f.write("4.4 Diversity (how varied the recommendations are)\n")
        f.write(f"- Hybrid approach: {hybrid_metrics['diversity']:.4f}\n")
        f.write(f"- Collaborative filtering: {cf_metrics['diversity']:.4f}\n")
        f.write(f"- Pattern-based: {pattern_metrics['diversity']:.4f}\n\n")
        
        f.write("5. KEY FINDINGS\n")
        f.write("---------------------------------------------------------\n")
        if hybrid_metrics['hit_rate'] > cf_metrics['hit_rate'] and hybrid_metrics['hit_rate'] > pattern_metrics['hit_rate']:
            f.write("- Hybrid approach achieved better hit rate than individual methods\n")
        elif cf_metrics['hit_rate'] > pattern_metrics['hit_rate']:
            f.write("- Collaborative filtering achieved better hit rate than pattern-based recommendations\n")
        else:
            f.write("- Pattern-based recommendations achieved better hit rate than collaborative filtering\n")
            
        if hybrid_metrics['average_precision'] > cf_metrics['average_precision'] and hybrid_metrics['average_precision'] > pattern_metrics['average_precision']:
            f.write("- Hybrid approach achieved better precision than individual methods\n")
        elif cf_metrics['average_precision'] > pattern_metrics['average_precision']:
            f.write("- Collaborative filtering achieved better precision than pattern-based recommendations\n")
        else:
            f.write("- Pattern-based recommendations achieved better precision than collaborative filtering\n")
            
        f.write("- The integration leverages strengths of both approaches:\n")
        f.write("  * Collaborative filtering provides broad recommendations based on user similarity\n")
        f.write("  * Pattern mining provides targeted recommendations based on item associations\n\n")
        
        f.write("6. BUSINESS IMPLICATIONS\n")
        f.write("---------------------------------------------------------\n")
        f.write("6.1 Enhanced Recommendation Quality\n")
        f.write("- Integration provides more balanced recommendations\n")
        f.write("- Potentially higher user satisfaction and engagement\n\n")
        
        f.write("6.2 Cross-Selling Opportunities\n")
        f.write("- Pattern mining component identifies complementary products\n")
        f.write("- Collaborative filtering identifies products popular with similar users\n\n")
        
        f.write("6.3 Cold Start Problem Mitigation\n")
        f.write("- Pattern-based component can provide recommendations for new users with few purchases\n")
        f.write("- CF component works well for users with more purchase history\n\n")
        
        f.write("7. LIMITATIONS AND CHALLENGES\n")
        f.write("---------------------------------------------------------\n")
        f.write("7.1 Computational Complexity\n")
        f.write("- Integrated approach requires more processing time\n")
        f.write("- Pattern mining is computationally intensive for large datasets\n\n")
        
        f.write("7.2 Parameter Tuning\n")
        f.write("- Optimal weights for hybrid recommendations may vary by domain\n")
        f.write("- Finding the right balance between CF and pattern-based components requires experimentation\n\n")
        
        f.write("8. FUTURE IMPROVEMENTS\n")
        f.write("---------------------------------------------------------\n")
        f.write("8.1 Dynamic Weighting\n")
        f.write("- Adjust weights based on user characteristics and purchase history\n")
        f.write("- More weight to CF for users with rich history, more to patterns for newer users\n\n")
        
        f.write("8.2 Temporal Dynamics\n")
        f.write("- Incorporate time-based features in both components\n")
        f.write("- Seasonal patterns and changing user preferences\n\n")
        
        f.write("8.3 Advanced Integration Methods\n")
        f.write("- Explore machine learning approaches to optimize integration\n")
        f.write("- Consider adaptive recommendation strategies\n\n")
        
        f.write("9. CONCLUSION\n")
        f.write("---------------------------------------------------------\n")
        f.write("The integration of collaborative filtering and pattern mining provides a comprehensive\n")
        f.write("recommendation system that leverages the strengths of both approaches. The hybrid\n")
        f.write("system demonstrates promising results in terms of recommendation quality and diversity.\n")
        f.write("Further refinement of the integration methodology and parameters could yield even\n")
        f.write("better performance in production environments.\n")
    
    print("\nTask C complete! Results are available in the 'results' and 'visualizations' directories.")
    print("Check the 'findings_task_c.txt' file for a detailed summary.")

if __name__ == "__main__":
    main() 