import os
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import random

from src.task_a.pattern_mining import PatternMiner
from src.task_b.collaborative_filtering import CollaborativeFilter
from src.task_c.integration import IntegratedSystem

def display_welcome_message():
    """Display welcome message and instructions to users"""
    print("="*80)
    print("Welcome to the Grocery Store Recommendation System")
    print("="*80)
    print("This system helps identify frequent purchase patterns and provides personalized recommendations.")
    print("Data source: Customer purchase history from the store's loyalty program.")
    print("="*80)

def load_datasets():
    """Load training and test datasets with error handling"""
    # Check if dataset files exist
    train_path = os.path.join('dataset', 'train.csv')
    test_path = os.path.join('dataset', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Dataset files not found!")
        print(f"Expected paths: {train_path} and {test_path}")
        sys.exit(1)
    
    # Load dataset
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"Successfully loaded training data: {train_data.shape[0]} records")
        print(f"Successfully loaded test data: {test_data.shape[0]} records")
        
        # Display dataset information
        print("\nTraining Data Preview:")
        print(train_data.head(3))
        print("\nMissing values in training data:", train_data.isnull().sum().sum())
        
        return train_data, test_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def get_valid_input(prompt, input_type=str, default=None, min_val=None, max_val=None):
    """Get valid user input with type checking and range validation"""
    default_msg = f" (default: {default})" if default is not None else ""
    while True:
        user_input = input(f"{prompt}{default_msg}: ")
        
        # Use default if no input provided
        if user_input == "" and default is not None:
            return default
        
        try:
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                value = user_input
                
            # Check range constraints
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}. Please try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}. Please try again.")
                continue
                
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")

def choose_user_id(data, message="Enter a user ID for recommendations"):
    """Allow user to choose a user ID or get a random one"""
    print("\nUser ID Selection:")
    print("1. Enter a specific user ID")
    print("2. Get a random user ID")
    
    choice = get_valid_input("Enter your choice", int, 2, 1, 2)
    
    if choice == 1:
        all_users = data['User_id'].unique()
        print(f"Available user IDs: Sample of 5 users - {all_users[:5]}")
        print(f"Total of {len(all_users)} unique users in the dataset")
        
        while True:
            user_id = get_valid_input(message, float)
            if user_id in all_users:
                return user_id
            else:
                print(f"User ID {user_id} not found in the dataset. Please try again.")
    else:
        all_users = data['User_id'].unique()
        user_id = random.choice(all_users)
        print(f"Randomly selected user ID: {user_id}")
        return user_id

def main():
    """Main entry point for the application"""
    display_welcome_message()
    
    # Load datasets
    train_data, test_data = load_datasets()
    
    # Create directories for outputs
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    while True:
        print("\n" + "="*50)
        print("SELECT A TASK TO RUN:")
        print("="*50)
        print("1. Task A: Pattern Mining")
        print("2. Task B: Collaborative Filtering")
        print("3. Task C: Integrated System")
        print("4. Task D: Interactive Analysis")
        print("5. Exit")
        
        choice = get_valid_input("\nEnter your choice", int, default=5, min_val=1, max_val=5)
        
        if choice == 1:
            run_task_a(train_data)
        elif choice == 2:
            run_task_b(train_data, test_data)
        elif choice == 3:
            run_task_c(train_data, test_data)
        elif choice == 4:
            run_interactive_analysis(train_data, test_data)
        elif choice == 5:
            print("\nThank you for using the Grocery Store Recommendation System. Goodbye!")
            break

def run_task_a(train_data):
    """Run Pattern Mining task with enhanced options"""
    print("\n" + "="*50)
    print("TASK A: PATTERN MINING")
    print("="*50)
    
    # Initialize pattern miner
    miner = PatternMiner(train_data)
    
    # Algorithm selection with validation
    print("\nSelect a mining algorithm:")
    print("1. Apriori (Slower but more interpretable)")
    print("2. FP Growth (Faster, better for large datasets)")
    
    algo_choice = get_valid_input("Enter your choice", int, default=2, min_val=1, max_val=2)
    algorithm = 'apriori' if algo_choice == 1 else 'fpgrowth'
    
    # Min support with validation
    min_support = get_valid_input(
        "Enter minimum support threshold (0.001-1.0)", 
        float, 
        default=0.01, 
        min_val=0.001, 
        max_val=1.0
    )
    
    # Min confidence for association rules
    min_confidence = get_valid_input(
        "Enter minimum confidence for association rules (0.1-1.0)", 
        float, 
        default=0.5, 
        min_val=0.1, 
        max_val=1.0
    )
    
    print(f"\nRunning {algorithm} algorithm with min_support={min_support} and min_confidence={min_confidence}")
    
    # Time the execution
    start_time = time.time()
    miner.run(algorithm=algorithm, min_support=min_support, min_confidence=min_confidence)
    execution_time = time.time() - start_time
    
    print(f"\nPattern mining completed in {execution_time:.2f} seconds.")
    print(f"Found {len(miner.frequent_itemsets)} frequent itemsets and {len(miner.rules)} association rules.")
    
    # Display results
    print("\nTop 10 frequent itemsets by support:")
    if hasattr(miner, 'frequent_itemsets') and miner.frequent_itemsets is not None:
        for i, (itemset, support) in enumerate(miner.frequent_itemsets.sort_values('support', ascending=False).head(10).iterrows()):
            items = ", ".join(list(itemset))
            print(f"{i+1}. {items} (Support: {support:.4f})")
    
    print("\nTop 10 association rules by confidence:")
    if hasattr(miner, 'rules') and miner.rules is not None and len(miner.rules) > 0:
        for i, row in miner.rules.sort_values('confidence', ascending=False).head(10).iterrows():
            antecedent = ", ".join(list(row['antecedents']))
            consequent = ", ".join(list(row['consequents']))
            print(f"{i+1}. {antecedent} → {consequent} (Conf: {row['confidence']:.4f}, Lift: {row['lift']:.4f})")
    else:
        print("No association rules found with the current parameters.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    miner.visualize_results()
    print("Visualizations saved to 'visualizations' directory.")
    
    # Ask if user wants to export results
    if get_valid_input("\nDo you want to export results to CSV? (y/n)", str, default='y').lower().startswith('y'):
        if hasattr(miner, 'frequent_itemsets') and miner.frequent_itemsets is not None:
            # Convert frozensets to strings for CSV export
            export_itemsets = miner.frequent_itemsets.copy()
            export_itemsets.index = export_itemsets.index.map(lambda x: ', '.join(x))
            export_itemsets.to_csv('results/frequent_itemsets.csv')
        
        if hasattr(miner, 'rules') and miner.rules is not None and len(miner.rules) > 0:
            # Convert frozensets to strings for CSV export
            export_rules = miner.rules.copy()
            export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(x))
            export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(x))
            export_rules.to_csv('results/association_rules.csv', index=False)
        
        print("Results exported to 'results' directory.")

def run_task_b(train_data, test_data):
    """Run Collaborative Filtering task with enhanced options"""
    print("\n" + "="*50)
    print("TASK B: COLLABORATIVE FILTERING")
    print("="*50)
    
    # Initialize collaborative filter
    cf = CollaborativeFilter(train_data)
    
    # Let user choose a specific user ID or get a random one
    user_id = choose_user_id(train_data)
    
    # Choose recommendation method
    print("\nSelect a recommendation method:")
    print("1. User-based collaborative filtering")
    print("2. Item-based collaborative filtering")
    print("3. SVD (Matrix Factorization)")
    print("4. Compare all methods")
    
    method_choice = get_valid_input("Enter your choice", int, default=4, min_val=1, max_val=4)
    method = {1: 'user', 2: 'item', 3: 'svd', 4: 'all'}[method_choice]
    
    # Number of recommendations
    n_recommendations = get_valid_input(
        "How many recommendations do you want?", 
        int, 
        default=10, 
        min_val=1, 
        max_val=50
    )
    
    # If using SVD, get number of components
    if method == 'svd' or method == 'all':
        n_components = get_valid_input(
            "Number of components for SVD", 
            int, 
            default=50, 
            min_val=10, 
            max_val=200
        )
    else:
        n_components = 50
    
    print("\nPreparing collaborative filtering model...")
    
    # Prepare data and compute similarities
    cf.prepare_data()
    
    if method in ['user', 'all']:
        print("Computing user-user similarity matrix...")
        cf.compute_similarity(mode='user')
    
    if method in ['item', 'all']:
        print("Computing item-item similarity matrix...")
        cf.compute_similarity(mode='item')
    
    if method in ['svd', 'all']:
        print(f"Training SVD model with {n_components} components...")
        cf.train_svd_model(n_components=n_components)
    
    # Generate recommendations
    print(f"\nGenerating recommendations for user {user_id}...")
    
    if method == 'all':
        start_time = time.time()
        user_recommendations = cf.generate_recommendations(
            user_id=user_id, method='user', n_recommendations=n_recommendations
        )
        user_time = time.time() - start_time
        
        start_time = time.time()
        item_recommendations = cf.generate_recommendations(
            user_id=user_id, method='item', n_recommendations=n_recommendations
        )
        item_time = time.time() - start_time
        
        start_time = time.time()
        svd_recommendations = cf.generate_recommendations(
            user_id=user_id, method='svd', n_recommendations=n_recommendations
        )
        svd_time = time.time() - start_time
        
        # Display all recommendations side by side
        print("\nComparison of Recommendation Methods:")
        print(f"{'User-Based':<30} | {'Item-Based':<30} | {'SVD-Based':<30}")
        print("-" * 90)
        
        for i in range(n_recommendations):
            user_rec = f"{list(user_recommendations.keys())[i]} ({list(user_recommendations.values())[i]:.4f})" if i < len(user_recommendations) else ""
            item_rec = f"{list(item_recommendations.keys())[i]} ({list(item_recommendations.values())[i]:.4f})" if i < len(item_recommendations) else ""
            svd_rec = f"{list(svd_recommendations.keys())[i]} ({list(svd_recommendations.values())[i]:.4f})" if i < len(svd_recommendations) else ""
            
            print(f"{user_rec:<30} | {item_rec:<30} | {svd_rec:<30}")
        
        print("\nExecution Times:")
        print(f"User-based method: {user_time:.4f} seconds")
        print(f"Item-based method: {item_time:.4f} seconds")
        print(f"SVD-based method: {svd_time:.4f} seconds")
        
        # Evaluate recommendations if test data is available
        if test_data is not None:
            print("\nEvaluating recommendation quality...")
            
            # Evaluate each method
            user_metrics = cf.evaluate_recommendations(test_data, method='user', n_users=5)
            item_metrics = cf.evaluate_recommendations(test_data, method='item', n_users=5)
            svd_metrics = cf.evaluate_recommendations(test_data, method='svd', n_users=5)
            
            # Display evaluation metrics
            print("\nQuality Metrics Comparison:")
            print(f"{'Metric':<15} | {'User-Based':<10} | {'Item-Based':<10} | {'SVD-Based':<10}")
            print("-" * 60)
            
            for metric in ['hit_rate', 'average_precision', 'coverage', 'diversity']:
                user_val = user_metrics[metric] if user_metrics else 0
                item_val = item_metrics[metric] if item_metrics else 0
                svd_val = svd_metrics[metric] if svd_metrics else 0
                
                print(f"{metric.capitalize():<15} | {user_val:.4f}{'*' if user_val > item_val and user_val > svd_val else '':<5} | {item_val:.4f}{'*' if item_val > user_val and item_val > svd_val else '':<5} | {svd_val:.4f}{'*' if svd_val > user_val and svd_val > item_val else '':<5}")
    else:
        # Generate recommendations for the chosen method
        start_time = time.time()
        recommendations = cf.generate_recommendations(
            user_id=user_id, method=method, n_recommendations=n_recommendations
        )
        execution_time = time.time() - start_time
        
        # Display recommendations
        print(f"\nTop {n_recommendations} recommendations using {method}-based method:")
        for i, (item, score) in enumerate(recommendations.items()):
            print(f"{i+1}. {item} (Score: {score:.4f})")
        
        print(f"\nRecommendations generated in {execution_time:.4f} seconds.")
        
        # Evaluate recommendations if test data is available
        if test_data is not None:
            print("\nEvaluating recommendation quality...")
            metrics = cf.evaluate_recommendations(test_data, method=method, n_users=5)
            
            if metrics:
                print("\nQuality Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    cf.visualize_results()
    print("Visualizations saved to 'visualizations' directory.")

def run_task_c(train_data, test_data):
    """Run Integrated System task with enhanced options"""
    print("\n" + "="*50)
    print("TASK C: INTEGRATED SYSTEM")
    print("="*50)
    
    # Initialize integrated system
    system = IntegratedSystem(train_data, test_data)
    
    # Let user choose a specific user ID or get a random one
    user_id = choose_user_id(train_data)
    
    # Pattern mining parameters
    print("\nPattern Mining Configuration:")
    min_support = get_valid_input(
        "Enter minimum support threshold (0.001-0.1)", 
        float, 
        default=0.005, 
        min_val=0.001, 
        max_val=0.1
    )
    
    algorithm = 'fpgrowth' if get_valid_input(
        "Use FP-Growth algorithm? (y/n)", 
        str, 
        default='y'
    ).lower().startswith('y') else 'apriori'
    
    # Collaborative filtering parameters
    print("\nCollaborative Filtering Configuration:")
    n_components = get_valid_input(
        "Number of components for SVD", 
        int, 
        default=50, 
        min_val=10, 
        max_val=200
    )
    
    # Integration parameters
    print("\nIntegration Configuration:")
    with_patterns = get_valid_input(
        "Generate hybrid recommendations with patterns? (y/n)", 
        str, 
        default='y'
    ).lower().startswith('y')
    
    n_recommendations = get_valid_input(
        "How many recommendations do you want?", 
        int, 
        default=10, 
        min_val=1, 
        max_val=50
    )
    
    # Run pattern mining
    print("\nRunning pattern mining...")
    start_time = time.time()
    system.mine_patterns(min_support=min_support, algorithm=algorithm)
    pattern_mining_time = time.time() - start_time
    
    # Prepare collaborative filtering
    print("\nPreparing collaborative filtering...")
    start_time = time.time()
    system.prepare_collaborative_filtering()
    cf_time = time.time() - start_time
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    
    if with_patterns:
        # Generate hybrid recommendations
        start_time = time.time()
        hybrid_recommendations = system.generate_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            with_patterns=True
        )
        hybrid_time = time.time() - start_time
        
        # Generate CF-only recommendations for comparison
        start_time = time.time()
        cf_recommendations = system.generate_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            with_patterns=False
        )
        cf_only_time = time.time() - start_time
        
        # Display execution time comparison
        print("\nExecution Time Comparison:")
        print(f"Pattern mining: {pattern_mining_time:.4f} seconds")
        print(f"Collaborative filtering setup: {cf_time:.4f} seconds")
        print(f"Hybrid recommendation generation: {hybrid_time:.4f} seconds")
        print(f"CF-only recommendation generation: {cf_only_time:.4f} seconds")
    else:
        # Generate CF-only recommendations
        start_time = time.time()
        cf_recommendations = system.generate_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            with_patterns=False
        )
        cf_only_time = time.time() - start_time
    
    # Evaluate recommendations
    if test_data is not None:
        print("\nEvaluating recommendation quality...")
        
        # How many users to evaluate
        n_eval_users = get_valid_input(
            "How many users to include in evaluation?", 
            int, 
            default=5, 
            min_val=1, 
            max_val=20
        )
        
        # Evaluate methods
        if with_patterns:
            print("\nEvaluating hybrid recommendations...")
            hybrid_metrics = system.evaluate_recommendations(
                method='hybrid', 
                n_users=n_eval_users
            )
            
            print("\nEvaluating collaborative filtering recommendations...")
            cf_metrics = system.evaluate_recommendations(
                method='cf', 
                n_users=n_eval_users
            )
            
            print("\nEvaluating pattern-based recommendations...")
            pattern_metrics = system.evaluate_recommendations(
                method='pattern', 
                n_users=n_eval_users
            )
            
            # Compare methods
            print("\nComparing recommendation methods...")
            comparison = system.compare_recommendation_methods(n_users=n_eval_users)
            
            if comparison is not None:
                print("\nRecommendation Methods Comparison:")
                print(comparison)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    system.visualize_results()
    
    # Export results
    print("\nExporting results...")
    system.export_results()
    
    print("\nTask C complete! Results are available in the 'results' and 'visualizations' directories.")
    print("Check the 'findings_task_c.txt' file for a detailed summary.")

def run_interactive_analysis(train_data, test_data):
    """Run Interactive Analysis with both pattern mining and collaborative filtering"""
    print("\n" + "="*50)
    print("TASK D: INTERACTIVE ANALYSIS")
    print("="*50)
    
    # Initialize components
    print("\nInitializing analysis components...")
    miner = PatternMiner(train_data)
    cf = CollaborativeFilter(train_data)
    system = IntegratedSystem(train_data, test_data)
    
    while True:
        print("\nInteractive Analysis Options:")
        print("1. Analyze frequent items for a specific user")
        print("2. Find users with similar purchase patterns")
        print("3. Discover complementary products")
        print("4. Compare recommendation methods for a user")
        print("5. Generate personalized shopping basket")
        print("6. Back to main menu")
        
        choice = get_valid_input("Enter your choice", int, default=6, min_val=1, max_val=6)
        
        if choice == 1:
            # Analyze frequent items for a specific user
            user_id = choose_user_id(train_data, "Select a user ID to analyze")
            
            # Get user's purchase history
            user_data = train_data[train_data['User_id'] == user_id]
            
            if len(user_data) == 0:
                print(f"No data found for user {user_id}")
                continue
            
            # Show user's purchase history
            user_items = user_data['itemDescription'].value_counts()
            
            print(f"\nUser {user_id} has purchased {len(user_items)} unique items:")
            for i, (item, count) in enumerate(user_items.items()):
                print(f"{i+1}. {item} ({count} times)")
            
            # Plot user's purchase history
            plt.figure(figsize=(12, 6))
            user_items.head(15).plot(kind='bar')
            plt.title(f"Top 15 Items Purchased by User {user_id}")
            plt.xlabel("Items")
            plt.ylabel("Purchase Count")
            plt.tight_layout()
            plt.savefig(f'visualizations/user_{user_id}_purchases.png')
            plt.close()
            
            print(f"\nVisualization saved to 'visualizations/user_{user_id}_purchases.png'")
            
        elif choice == 2:
            # Find users with similar purchase patterns
            user_id = choose_user_id(train_data, "Select a user ID to find similar users")
            
            # Prepare data
            if not hasattr(cf, 'user_similarity') or cf.user_similarity is None:
                print("Computing user similarity matrix...")
                cf.prepare_data()
                cf.compute_similarity(mode='user')
            
            # Find similar users
            if user_id in cf.user_similarity:
                similar_users = cf.user_similarity[user_id].sort_values(ascending=False)
                
                # Exclude the user itself
                similar_users = similar_users.drop(user_id, errors='ignore')
                
                print(f"\nTop 10 similar users to User {user_id}:")
                for i, (similar_user, similarity) in enumerate(similar_users.head(10).items()):
                    print(f"{i+1}. User {similar_user} (Similarity: {similarity:.4f})")
                
                # Get purchase history of the most similar user
                if len(similar_users) > 0:
                    most_similar = similar_users.index[0]
                    similar_user_data = train_data[train_data['User_id'] == most_similar]
                    similar_items = similar_user_data['itemDescription'].value_counts()
                    
                    print(f"\nMost similar user ({most_similar}) has purchased:")
                    for i, (item, count) in enumerate(similar_items.head(10).items()):
                        print(f"{i+1}. {item} ({count} times)")
            else:
                print(f"User {user_id} not found in the similarity matrix.")
            
        elif choice == 3:
            # Discover complementary products
            # Run pattern mining first
            if not hasattr(miner, 'rules') or miner.rules is None or len(miner.rules) == 0:
                print("Running pattern mining to discover product associations...")
                miner.run(algorithm='fpgrowth', min_support=0.005, min_confidence=0.3)
            
            # Allow user to select a product
            item_choices = sorted(train_data['itemDescription'].unique())
            
            print("\nSelect a product to find complementary items:")
            for i, item in enumerate(item_choices[:20]):
                print(f"{i+1}. {item}")
            
            if len(item_choices) > 20:
                print(f"... and {len(item_choices) - 20} more items")
                
            # Let user enter a search term
            search_term = get_valid_input("Enter a product name or search term", str)
            matching_items = [item for item in item_choices if search_term.lower() in item.lower()]
            
            if len(matching_items) == 0:
                print(f"No items found matching '{search_term}'")
                continue
            
            # Display matching items
            print("\nMatching products:")
            for i, item in enumerate(matching_items):
                print(f"{i+1}. {item}")
            
            item_idx = get_valid_input(
                "Select a product number", 
                int, 
                default=1, 
                min_val=1, 
                max_val=len(matching_items)
            )
            selected_item = matching_items[item_idx - 1]
            
            # Find rules where the selected item is in antecedent
            if hasattr(miner, 'rules') and miner.rules is not None and len(miner.rules) > 0:
                # Find rules containing the selected item
                complementary_rules = []
                
                for _, rule in miner.rules.iterrows():
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    
                    if selected_item in antecedents:
                        complementary_rules.append({
                            'antecedents': antecedents,
                            'consequents': consequents,
                            'confidence': rule['confidence'],
                            'lift': rule['lift']
                        })
                
                if len(complementary_rules) > 0:
                    print(f"\nComplementary products for '{selected_item}':")
                    
                    # Sort by confidence
                    complementary_rules.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    for i, rule in enumerate(complementary_rules[:10]):
                        antecedents = ", ".join([item for item in rule['antecedents'] if item != selected_item])
                        consequents = ", ".join(rule['consequents'])
                        
                        if antecedents and consequents:
                            print(f"{i+1}. When buying {selected_item} and {antecedents}, customers also buy {consequents}")
                            print(f"   Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")
                        elif consequents:
                            print(f"{i+1}. When buying {selected_item}, customers also buy {consequents}")
                            print(f"   Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")
                else:
                    print(f"No complementary products found for '{selected_item}'")
            else:
                print("No association rules available. Please run pattern mining first.")
            
        elif choice == 4:
            # Compare recommendation methods for a user
            user_id = choose_user_id(train_data, "Select a user ID for recommendation comparison")
            
            # Initialize integrated system if not already done
            print("\nPreparing recommendation systems...")
            
            # Run with minimal parameters for speed
            system.mine_patterns(min_support=0.005, algorithm='fpgrowth')
            system.prepare_collaborative_filtering()
            
            # Generate recommendations using different methods
            print("\nGenerating recommendations using different methods...")
            
            # Generate hybrid recommendations
            hybrid_recommendations = system.generate_recommendations(
                user_id=user_id,
                n_recommendations=10,
                with_patterns=True
            )
            
            # Generate CF-only recommendations
            cf_recommendations = system.collaborative_filter.generate_recommendations(
                user_id=user_id,
                method='svd',
                n_recommendations=10
            )
            
            # Generate pattern-based recommendations
            user_data = train_data[train_data['User_id'] == user_id]
            user_items = set(user_data['itemDescription'].unique())
            
            # Score patterns for this user
            scored_patterns = system.pattern_miner.score_patterns(user_id=user_id)
            
            # Get recommendations based on patterns
            pattern_recommendations = system._get_pattern_based_recommendations(
                user_id=user_id,
                scored_patterns=scored_patterns,
                user_items=user_items,
                n_recommendations=10
            )
            
            # Compare methods
            print("\nRecommendation Methods Comparison:")
            
            max_len = max(len(hybrid_recommendations), len(cf_recommendations), len(pattern_recommendations))
            
            print(f"{'#':<3} | {'Hybrid':<30} | {'Collaborative Filtering':<30} | {'Pattern-Based':<30}")
            print("-" * 100)
            
            for i in range(max_len):
                hybrid_rec = f"{list(hybrid_recommendations.keys())[i]} ({list(hybrid_recommendations.values())[i]:.4f})" if i < len(hybrid_recommendations) else ""
                cf_rec = f"{list(cf_recommendations.keys())[i]} ({list(cf_recommendations.values())[i]:.4f})" if i < len(cf_recommendations) else ""
                pattern_rec = f"{list(pattern_recommendations.keys())[i]} ({list(pattern_recommendations.values())[i]:.4f})" if i < len(pattern_recommendations) else ""
                
                print(f"{i+1:<3} | {hybrid_rec:<30} | {cf_rec:<30} | {pattern_rec:<30}")
            
            # Generate visualization
            print("\nGenerating visualization...")
            system.visualize_results()
            print("Visualization saved to 'visualizations' directory.")
            
        elif choice == 5:
            # Generate personalized shopping basket
            user_id = choose_user_id(train_data, "Select a user ID for shopping basket generation")
            
            # Initialize integrated system if not already done
            print("\nPreparing recommendation systems...")
            
            if not hasattr(system.pattern_miner, 'frequent_itemsets') or system.pattern_miner.frequent_itemsets is None:
                system.mine_patterns(min_support=0.005, algorithm='fpgrowth')
            
            if not hasattr(system.collaborative_filter, 'user_item_matrix') or system.collaborative_filter.user_item_matrix is None:
                system.prepare_collaborative_filtering()
            
            # Get user's purchase history
            user_data = train_data[train_data['User_id'] == user_id]
            user_items = set(user_data['itemDescription'].unique())
            
            # Generate recommendations
            print("\nGenerating personalized shopping basket...")
            
            # Get hybrid recommendations
            hybrid_recommendations = system.generate_recommendations(
                user_id=user_id,
                n_recommendations=15,
                with_patterns=True
            )
            
            # Categorize items (this is a simple categorization for demonstration)
            categories = {
                'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
                'meat': ['meat', 'chicken', 'beef', 'pork', 'turkey', 'ham', 'sausage'],
                'produce': ['vegetables', 'fruit', 'salad', 'onions', 'potatoes'],
                'bakery': ['bread', 'rolls', 'buns', 'pastry', 'cake'],
                'beverages': ['water', 'soda', 'juice', 'beer', 'wine', 'coffee', 'tea'],
                'snacks': ['chips', 'chocolate', 'candy', 'snack', 'nuts', 'popcorn'],
                'household': ['cleaner', 'detergent', 'paper', 'napkins', 'toilet', 'tissue']
            }
            
            # Categorize recommendations
            categorized_items = {category: [] for category in categories.keys()}
            categorized_items['other'] = []
            
            for item, score in hybrid_recommendations.items():
                assigned = False
                for category, keywords in categories.items():
                    if any(keyword in item.lower() for keyword in keywords):
                        categorized_items[category].append((item, score))
                        assigned = True
                        break
                
                if not assigned:
                    categorized_items['other'].append((item, score))
            
            # Display personalized shopping basket
            print(f"\nPersonalized Shopping Basket for User {user_id}:")
            print(f"Based on your shopping history of {len(user_items)} items")
            
            for category, items in categorized_items.items():
                if len(items) > 0:
                    print(f"\n{category.capitalize()} Section:")
                    for item, score in items:
                        print(f"- {item} (Recommendation Score: {score:.4f})")
            
            # Create a shopping list file
            with open(f'results/shopping_list_user_{user_id}.txt', 'w') as f:
                f.write(f"PERSONALIZED SHOPPING LIST FOR USER {user_id}\n")
                f.write("="*50 + "\n\n")
                
                f.write("ITEMS TO CONSIDER:\n\n")
                
                for category, items in categorized_items.items():
                    if len(items) > 0:
                        f.write(f"{category.capitalize()} Section:\n")
                        for item, score in items:
                            f.write(f"- {item}\n")
                        f.write("\n")
            
            print(f"\nShopping list saved to 'results/shopping_list_user_{user_id}.txt'")
            
        elif choice == 6:
            # Back to main menu
            break

if __name__ == "__main__":
    main() 