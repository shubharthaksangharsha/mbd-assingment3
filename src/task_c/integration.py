import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import random
from collections import defaultdict

# Add parent directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components from Task A and Task B
from src.task_a.pattern_mining import PatternMiner
from src.task_b.collaborative_filtering import CollaborativeFilter
from utils.data_utils import (
    preprocess_data,
    get_user_transactions,
    get_all_users,
    get_all_items
)

class IntegratedSystem:
    """
    Integrated system that combines pattern mining and collaborative filtering
    for enhanced recommendations.
    """
    
    def __init__(self, train_data, test_data=None):
        """
        Initialize the integrated system
        
        Args:
            train_data (pd.DataFrame): Training data
            test_data (pd.DataFrame, optional): Test data
        """
        self.train_data = train_data
        self.test_data = test_data
        self.data = preprocess_data(train_data)
        
        # Initialize components
        self.pattern_miner = PatternMiner(train_data)
        self.collaborative_filter = CollaborativeFilter(train_data)
        
        # Store recommendations
        self.cf_recommendations = None
        self.pattern_recommendations = None
        self.hybrid_recommendations = None
        self.user_id = None
        
        # Create directories for outputs
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def mine_patterns(self, min_support=0.005, algorithm='fpgrowth'):
        """
        Run pattern mining
        
        Args:
            min_support (float): Minimum support threshold
            algorithm (str): Mining algorithm ('apriori' or 'fpgrowth')
        """
        print(f"Running pattern mining with {algorithm} algorithm...")
        # Lower the minimum confidence to 0.2 instead of the default 0.5
        # This will generate more association rules from the frequent itemsets
        self.pattern_miner.run(algorithm=algorithm, min_support=min_support, min_confidence=0.2, min_lift=0.5)
        print("Pattern mining completed.")
    
    def prepare_collaborative_filtering(self):
        """Prepare the collaborative filtering component"""
        print("Preparing collaborative filtering...")
        self.collaborative_filter.prepare_data()
        
        # Compute similarity matrices
        self.collaborative_filter.compute_similarity(mode='user')
        self.collaborative_filter.compute_similarity(mode='item')
        
        # Train SVD model
        self.collaborative_filter.train_svd_model(n_components=50)
        
        print("Collaborative filtering preparation completed.")
    
    def generate_recommendations(self, user_id=None, n_recommendations=10, with_patterns=True):
        """
        Generate recommendations using the integrated system
        
        Args:
            user_id (int, optional): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to generate
            with_patterns (bool): Whether to incorporate pattern mining results
            
        Returns:
            dict: Hybrid recommendations with scores
        """
        # If user_id is not provided, select a random user
        if user_id is None:
            all_users = get_all_users(self.data)
            user_id = random.choice(all_users)
            print(f"No user ID provided. Selected random user: {user_id}")
        
        self.user_id = user_id
        
        # Get user's purchase history
        user_data = get_user_transactions(self.data, user_id)
        user_items = set(user_data['itemDescription'].unique())
        
        print(f"User {user_id} has purchased {len(user_items)} unique items")
        
        # 1. Get collaborative filtering recommendations (using SVD)
        print("\nGenerating collaborative filtering recommendations...")
        self.cf_recommendations = self.collaborative_filter.generate_recommendations(
            user_id=user_id,
            method='svd',
            n_recommendations=n_recommendations * 2  # Get more recommendations to allow for filtering
        )
        
        # 2. If incorporating patterns, run pattern mining and get pattern-based recommendations
        if with_patterns:
            print("\nGenerating pattern-based recommendations...")
            
            # Ensure patterns are mined
            if not hasattr(self.pattern_miner, 'frequent_itemsets') or self.pattern_miner.frequent_itemsets is None:
                self.mine_patterns()
            
            # Score patterns for this user
            scored_patterns = self.pattern_miner.score_patterns(user_id=user_id)
            
            # Get recommendations based on patterns
            self.pattern_recommendations = self._get_pattern_based_recommendations(
                user_id=user_id,
                scored_patterns=scored_patterns,
                user_items=user_items,
                n_recommendations=n_recommendations * 2  # Get more to allow for filtering
            )
            
            # 3. Generate hybrid recommendations
            print("\nGenerating hybrid recommendations...")
            self.hybrid_recommendations = self._generate_hybrid_recommendations(
                cf_recommendations=self.cf_recommendations,
                pattern_recommendations=self.pattern_recommendations,
                n_recommendations=n_recommendations
            )
            
            # Print the hybrid recommendations
            print("\nTop hybrid recommendations:")
            for i, (item, score) in enumerate(self.hybrid_recommendations.items()):
                print(f"{i+1}. {item} (Score: {score:.4f})")
            
            return self.hybrid_recommendations
        else:
            # If not using patterns, just return the CF recommendations
            print("\nPattern mining not incorporated. Using only collaborative filtering recommendations.")
            
            # Print the CF recommendations
            print("\nTop recommendations (collaborative filtering only):")
            for i, (item, score) in enumerate(self.cf_recommendations.items()):
                print(f"{i+1}. {item} (Score: {score:.4f})")
            
            return self.cf_recommendations
    
    def _get_pattern_based_recommendations(self, user_id, scored_patterns, user_items, n_recommendations=10):
        """
        Generate recommendations based on frequent patterns
        
        Args:
            user_id (int): User ID
            scored_patterns (pd.DataFrame): Scored patterns from pattern mining
            user_items (set): Items the user has already purchased
            n_recommendations (int): Number of recommendations
            
        Returns:
            dict: Pattern-based recommendations with scores
        """
        # Get all items from the frequent itemsets
        all_pattern_items = set()
        for itemset in scored_patterns['itemsets']:
            all_pattern_items.update(itemset)
        
        # Filter out items the user has already purchased
        new_items = all_pattern_items - user_items
        
        # If we have association rules, use them first for recommendations
        if hasattr(self.pattern_miner, 'association_rules') and len(self.pattern_miner.association_rules) > 0:
            user_item_rules = {}
            # Look for rules where items the user has purchased are in the antecedent
            for _, rule in self.pattern_miner.association_rules.iterrows():
                antecedent_items = set(rule['antecedents'])
                consequent_items = set(rule['consequents'])
                
                # Check if there's an overlap between user's items and rule antecedents
                if len(antecedent_items.intersection(user_items)) > 0:
                    # For each item in the consequent that the user hasn't purchased
                    for item in consequent_items - user_items:
                        if item not in user_item_rules:
                            user_item_rules[item] = rule['confidence'] * rule['lift']  # Score by confidence and lift
                        else:
                            # Take the max score if item appears in multiple rules
                            user_item_rules[item] = max(user_item_rules[item], rule['confidence'] * rule['lift'])
            
            # If we found rules relevant to this user, use them
            if user_item_rules:
                print(f"Found {len(user_item_rules)} items from association rules for user {user_id}")
                # Get top items from rules
                rule_recommendations = dict(sorted(user_item_rules.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
                
                # If we don't have enough recommendations from rules, supplement with pattern-based
                if len(rule_recommendations) < n_recommendations:
                    # Continue with pattern-based scoring for remaining items
                    pass
                else:
                    return rule_recommendations
        
        # Score each item based on the patterns it appears in
        item_scores = defaultdict(float)
        
        for idx, row in scored_patterns.iterrows():
            itemset = set(row['itemsets'])
            
            # Only consider patterns that have at least one item the user has purchased
            if len(itemset.intersection(user_items)) > 0:
                # Score items in this pattern that the user hasn't purchased yet
                for item in itemset - user_items:
                    # Weight by support and user relevance (if available)
                    if 'user_relevance' in row:
                        item_scores[item] += row['support'] * row['user_relevance'] * row['length']
                    else:
                        item_scores[item] += row['support'] * row['length']
        
        # If we already have some recommendations from rules, add the remaining spots
        if 'rule_recommendations' in locals() and rule_recommendations:
            remaining_spots = n_recommendations - len(rule_recommendations)
            # Filter out items already in rule_recommendations
            item_scores = {k: v for k, v in item_scores.items() if k not in rule_recommendations}
            # Get top items for remaining spots
            pattern_recommendations = dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:remaining_spots])
            # Combine recommendations
            combined_recommendations = {**rule_recommendations, **pattern_recommendations}
            recommendations = combined_recommendations
        else:
            # Sort by score and get top N
            recommendations = dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
        
        # Print the pattern-based recommendations
        print("\nTop pattern-based recommendations:")
        for i, (item, score) in enumerate(recommendations.items()):
            print(f"{i+1}. {item} (Score: {score:.4f})")
        
        return recommendations
    
    def _generate_hybrid_recommendations(self, cf_recommendations, pattern_recommendations, n_recommendations=10):
        """
        Generate hybrid recommendations by combining CF and pattern-based recommendations
        
        Args:
            cf_recommendations (dict): Collaborative filtering recommendations
            pattern_recommendations (dict): Pattern-based recommendations
            n_recommendations (int): Number of recommendations
            
        Returns:
            dict: Hybrid recommendations with scores
        """
        # Normalize scores for each method
        cf_max = max(cf_recommendations.values()) if cf_recommendations else 1
        pattern_max = max(pattern_recommendations.values()) if pattern_recommendations else 1
        
        normalized_cf = {item: score/cf_max for item, score in cf_recommendations.items()}
        normalized_pattern = {item: score/pattern_max for item, score in pattern_recommendations.items()}
        
        # Check if pattern recommendations are diverse enough
        pattern_weight = 0.4  # Default pattern weight
        
        # If pattern recommendations have more than 5 unique items, increase their weight
        if len(pattern_recommendations) >= 5:
            pattern_weight = 0.5
            print(f"Increasing pattern recommendations weight to {pattern_weight} due to good diversity")
        
        # Calculate the CF weight accordingly
        cf_weight = 1.0 - pattern_weight
        
        # Combine recommendations with weighting
        # Items that appear in both get a boost
        hybrid_scores = defaultdict(float)
        
        # Add CF scores
        for item, score in normalized_cf.items():
            hybrid_scores[item] += cf_weight * score
        
        # Add pattern scores
        for item, score in normalized_pattern.items():
            hybrid_scores[item] += pattern_weight * score
        
        # Higher boost (30%) for items that appear in both recommendation sets
        for item in set(normalized_cf.keys()).intersection(set(normalized_pattern.keys())):
            hybrid_scores[item] *= 1.3  # 30% boost
        
        # Sort and get top N
        recommendations = dict(sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
        
        return recommendations
    
    def evaluate_recommendations(self, method='hybrid', n_users=50):
        """
        Evaluate recommendation quality using test data
        
        Args:
            method (str): Recommendation method ('hybrid', 'cf', or 'pattern')
            n_users (int): Number of users to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if self.test_data is None:
            print("Warning: No test data provided. Unable to evaluate recommendations.")
            return None
        
        # Clone the test data to avoid modifying the original
        test_data_clone = self.test_data.copy()
        
        # Ensure we have the User_id column before preprocessing
        if 'User_id' not in test_data_clone.columns:
            # Try to find variant spellings
            user_id_variants = ['user_id', 'userId', 'userid', 'user id', 'id']
            for variant in user_id_variants:
                if variant in test_data_clone.columns:
                    print(f"Found variant column name '{variant}', renaming to 'User_id'")
                    test_data_clone.rename(columns={variant: 'User_id'}, inplace=True)
                    break
        
        # Save the User_id column before preprocessing (very important!)
        user_ids = None
        if 'User_id' in test_data_clone.columns:
            # Create a mapping of original row indices to User_ids
            user_ids = {i: uid for i, uid in enumerate(test_data_clone['User_id'])}
            print(f"Saved {len(user_ids)} User_id values before preprocessing")
        else:
            print("Warning: Could not find User_id column or any variants in test data")
            return {
                'hit_rate': 0,
                'average_precision': 0,
                'coverage': 0,
                'diversity': 0
            }
        
        # Preprocess the test data
        test_data = preprocess_data(test_data_clone)
        
        # Re-add the User_id column if it was lost during preprocessing
        if 'User_id' not in test_data.columns and user_ids is not None:
            print("Restoring User_id column after preprocessing")
            # Create a new User_id column based on the original row indices
            test_data['User_id'] = test_data.index.map(lambda i: user_ids.get(i, None))
            
            # Drop rows that couldn't be mapped back to a User_id
            before_len = len(test_data)
            test_data = test_data.dropna(subset=['User_id'])
            after_len = len(test_data)
            if before_len != after_len:
                print(f"Dropped {before_len - after_len} rows with missing User_id values")
        
        if 'User_id' not in test_data.columns:
            print("Error: Unable to preserve or reconstruct the User_id column in test data.")
            return {
                'hit_rate': 0,
                'average_precision': 0,
                'coverage': 0,
                'diversity': 0
            }
        
        # Get users that exist in both train and test
        train_users = set(self.data['User_id'].unique())
        test_users = set(test_data['User_id'].unique())
        common_users = list(train_users.intersection(test_users))
        
        if len(common_users) == 0:
            print("Error: No common users found between train and test data.")
            return {
                'hit_rate': 0,
                'average_precision': 0,
                'coverage': 0,
                'diversity': 0
            }
        
        # If there are fewer common users than requested, use all common users
        n_users = min(n_users, len(common_users))
        
        # Randomly select users for evaluation
        eval_users = random.sample(common_users, n_users)
        
        print(f"Evaluating {method} recommendations for {n_users} users...")
        
        # Initialize evaluation metrics
        metrics = {
            'hit_rate': 0,
            'average_precision': 0,
            'coverage': set(),
            'diversity': 0
        }
        
        # Dictionary to count recommendation co-occurrences (for diversity calculation)
        item_co_occurrences = defaultdict(int)
        
        # For each user
        successful_evaluations = 0
        for user_id in eval_users:
            # Get user's actual purchases in test data
            user_test_data = test_data[test_data['User_id'] == user_id]
            actual_items = set(user_test_data['itemDescription'].unique())
            
            # If user has no purchases in test data, skip
            if len(actual_items) == 0:
                continue
            
            try:
                # Generate recommendations for the user
                if method == 'hybrid':
                    recommendations = self.generate_recommendations(
                        user_id=user_id,
                        n_recommendations=10,
                        with_patterns=True
                    )
                elif method == 'cf':
                    recommendations = self.collaborative_filter.generate_recommendations(
                        user_id=user_id,
                        method='svd',
                        n_recommendations=10
                    )
                elif method == 'pattern':
                    user_data = get_user_transactions(self.data, user_id)
                    user_items = set(user_data['itemDescription'].unique())
                    
                    # Ensure patterns are mined
                    if not hasattr(self.pattern_miner, 'frequent_itemsets') or self.pattern_miner.frequent_itemsets is None:
                        self.mine_patterns()
                    
                    # Score patterns for this user
                    scored_patterns = self.pattern_miner.score_patterns(user_id=user_id)
                    
                    # Get recommendations based on patterns
                    recommendations = self._get_pattern_based_recommendations(
                        user_id=user_id,
                        scored_patterns=scored_patterns,
                        user_items=user_items,
                        n_recommendations=10
                    )
                else:
                    raise ValueError("Method must be 'hybrid', 'cf', or 'pattern'")
                
                recommended_items = list(recommendations.keys())
                
                # Update all recommended items set
                metrics['coverage'].update(recommended_items)
                
                # Update co-occurrence counts for diversity calculation
                for i, item1 in enumerate(recommended_items):
                    for item2 in recommended_items[i+1:]:
                        if item1 < item2:
                            item_co_occurrences[(item1, item2)] += 1
                        else:
                            item_co_occurrences[(item2, item1)] += 1
                
                # Calculate hit rate (proportion of recommended items that were actually purchased)
                hits = [item for item in recommended_items if item in actual_items]
                hit_rate = len(hits) / len(actual_items) if len(actual_items) > 0 else 0
                metrics['hit_rate'] += hit_rate
                
                # Calculate average precision
                precision = len(hits) / len(recommended_items) if len(recommended_items) > 0 else 0
                metrics['average_precision'] += precision
                
                successful_evaluations += 1
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        if successful_evaluations == 0:
            print("Error: No users were successfully evaluated.")
            return {
                'hit_rate': 0,
                'average_precision': 0,
                'coverage': 0,
                'diversity': 0
            }
        
        # Calculate average metrics
        metrics['hit_rate'] /= successful_evaluations
        metrics['average_precision'] /= successful_evaluations
        
        # Calculate coverage (proportion of all items that were recommended)
        all_items = set(self.data['itemDescription'].unique())
        metrics['coverage'] = len(metrics['coverage']) / len(all_items) if len(all_items) > 0 else 0
        
        # Calculate diversity (inverse of average co-occurrence)
        if len(item_co_occurrences) > 0:
            avg_co_occurrence = sum(item_co_occurrences.values()) / len(item_co_occurrences)
            metrics['diversity'] = 1 / (1 + avg_co_occurrence)
        else:
            metrics['diversity'] = 1.0
        
        print("\nEvaluation metrics:")
        print(f"Hit rate: {metrics['hit_rate']:.4f}")
        print(f"Average precision: {metrics['average_precision']:.4f}")
        print(f"Coverage: {metrics['coverage']:.4f}")
        print(f"Diversity: {metrics['diversity']:.4f}")
        
        return metrics
    
    def compare_recommendation_methods(self, n_users=50):
        """
        Compare different recommendation methods
        
        Args:
            n_users (int): Number of users to evaluate
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if self.test_data is None:
            print("Warning: No test data provided. Unable to compare recommendation methods.")
            return None
        
        print("\nComparing recommendation methods...")
        
        # Evaluate each method
        try:
            print("\nEvaluating hybrid recommendations...")
            hybrid_metrics = self.evaluate_recommendations(method='hybrid', n_users=n_users)
        except Exception as e:
            print(f"Error evaluating hybrid recommendations: {e}")
            hybrid_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
        
        try:
            print("\nEvaluating collaborative filtering recommendations...")
            cf_metrics = self.evaluate_recommendations(method='cf', n_users=n_users)
        except Exception as e:
            print(f"Error evaluating collaborative filtering recommendations: {e}")
            cf_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
        
        try:
            print("\nEvaluating pattern-based recommendations...")
            pattern_metrics = self.evaluate_recommendations(method='pattern', n_users=n_users)
        except Exception as e:
            print(f"Error evaluating pattern-based recommendations: {e}")
            pattern_metrics = {'hit_rate': 0, 'average_precision': 0, 'coverage': 0, 'diversity': 0}
        
        # Combine results
        comparison = pd.DataFrame({
            'Method': ['Hybrid', 'Collaborative Filtering', 'Pattern-based'],
            'Hit Rate': [
                hybrid_metrics['hit_rate'] if hybrid_metrics else 0,
                cf_metrics['hit_rate'] if cf_metrics else 0,
                pattern_metrics['hit_rate'] if pattern_metrics else 0
            ],
            'Precision': [
                hybrid_metrics['average_precision'] if hybrid_metrics else 0,
                cf_metrics['average_precision'] if cf_metrics else 0,
                pattern_metrics['average_precision'] if pattern_metrics else 0
            ],
            'Coverage': [
                hybrid_metrics['coverage'] if hybrid_metrics else 0,
                cf_metrics['coverage'] if cf_metrics else 0,
                pattern_metrics['coverage'] if pattern_metrics else 0
            ],
            'Diversity': [
                hybrid_metrics['diversity'] if hybrid_metrics else 0,
                cf_metrics['diversity'] if cf_metrics else 0,
                pattern_metrics['diversity'] if pattern_metrics else 0
            ]
        })
        
        # Save comparison to CSV
        comparison.to_csv('results/recommendation_methods_comparison.csv', index=False)
        
        try:
            # Create bar chart
            plt.figure(figsize=(14, 10))
            
            # Hit Rate
            plt.subplot(2, 2, 1)
            plt.bar(comparison['Method'], comparison['Hit Rate'])
            plt.title('Hit Rate Comparison')
            plt.ylabel('Hit Rate')
            
            # Precision
            plt.subplot(2, 2, 2)
            plt.bar(comparison['Method'], comparison['Precision'])
            plt.title('Precision Comparison')
            plt.ylabel('Precision')
            
            # Coverage
            plt.subplot(2, 2, 3)
            plt.bar(comparison['Method'], comparison['Coverage'])
            plt.title('Coverage Comparison')
            plt.ylabel('Coverage')
            
            # Diversity
            plt.subplot(2, 2, 4)
            plt.bar(comparison['Method'], comparison['Diversity'])
            plt.title('Diversity Comparison')
            plt.ylabel('Diversity')
            
            plt.tight_layout()
            plt.savefig('visualizations/recommendation_methods_comparison.png')
            plt.close()
            
            print("Comparison visualization saved to 'visualizations/recommendation_methods_comparison.png'")
        except Exception as e:
            print(f"Error creating comparison visualization: {e}")
        
        return comparison
    
    def visualize_results(self):
        """Generate visualizations of the recommendation results"""
        if self.user_id is None:
            print("No recommendations available to visualize. Generate recommendations first.")
            return
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Compare recommendation methods for the current user
        if self.cf_recommendations and self.pattern_recommendations and self.hybrid_recommendations:
            # Get top 5 from each method
            cf_top5 = dict(list(self.cf_recommendations.items())[:5])
            pattern_top5 = dict(list(self.pattern_recommendations.items())[:5])
            hybrid_top5 = dict(list(self.hybrid_recommendations.items())[:5])
            
            # Plot comparison
            plt.figure(figsize=(15, 10))
            
            # CF recommendations
            plt.subplot(3, 1, 1)
            plt.barh(list(cf_top5.keys()), list(cf_top5.values()))
            plt.title(f'Top 5 Collaborative Filtering Recommendations for User {self.user_id}')
            plt.xlabel('Score')
            
            # Pattern recommendations
            plt.subplot(3, 1, 2)
            plt.barh(list(pattern_top5.keys()), list(pattern_top5.values()))
            plt.title(f'Top 5 Pattern-Based Recommendations for User {self.user_id}')
            plt.xlabel('Score')
            
            # Hybrid recommendations
            plt.subplot(3, 1, 3)
            plt.barh(list(hybrid_top5.keys()), list(hybrid_top5.values()))
            plt.title(f'Top 5 Hybrid Recommendations for User {self.user_id}')
            plt.xlabel('Score')
            
            plt.tight_layout()
            plt.savefig('visualizations/recommendation_methods_for_user.png')
            plt.close()
        
        # 2. Visualize user purchase history
        user_data = get_user_transactions(self.data, self.user_id)
        
        if len(user_data) > 0:
            user_items = user_data['itemDescription'].value_counts().head(10)
            
            plt.figure(figsize=(12, 6))
            plt.barh(user_items.index, user_items.values)
            plt.title(f'Top 10 Purchased Items by User {self.user_id}')
            plt.xlabel('Purchase Count')
            plt.tight_layout()
            plt.savefig('visualizations/user_purchase_history.png')
            plt.close()
        
        # 3. Recommendation overlap visualization
        if self.cf_recommendations and self.pattern_recommendations and self.hybrid_recommendations:
            cf_items = set(self.cf_recommendations.keys())
            pattern_items = set(self.pattern_recommendations.keys())
            hybrid_items = set(self.hybrid_recommendations.keys())
            
            from matplotlib_venn import venn3
            
            plt.figure(figsize=(10, 10))
            venn3([cf_items, pattern_items, hybrid_items], 
                 ('Collaborative Filtering', 'Pattern-Based', 'Hybrid'))
            plt.title(f'Overlap of Recommendations for User {self.user_id}')
            plt.savefig('visualizations/recommendation_overlap.png')
            plt.close()
        
        print("Visualizations saved to 'visualizations' directory.")
    
    def export_results(self):
        """Export results to CSV files"""
        if self.user_id is None:
            print("No recommendations available to export. Generate recommendations first.")
            return
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Export recommendations
        if self.cf_recommendations:
            cf_df = pd.DataFrame({
                'item': list(self.cf_recommendations.keys()),
                'score': list(self.cf_recommendations.values())
            })
            cf_df.to_csv(f'results/cf_recommendations_user_{self.user_id}.csv', index=False)
        
        if self.pattern_recommendations:
            pattern_df = pd.DataFrame({
                'item': list(self.pattern_recommendations.keys()),
                'score': list(self.pattern_recommendations.values())
            })
            pattern_df.to_csv(f'results/pattern_recommendations_user_{self.user_id}.csv', index=False)
        
        if self.hybrid_recommendations:
            hybrid_df = pd.DataFrame({
                'item': list(self.hybrid_recommendations.keys()),
                'score': list(self.hybrid_recommendations.values())
            })
            hybrid_df.to_csv(f'results/hybrid_recommendations_user_{self.user_id}.csv', index=False)
        
        # Create summary text file
        with open(f'results/integrated_system_summary.txt', 'w') as f:
            f.write("==========================================================\n")
            f.write("INTEGRATED RECOMMENDATION SYSTEM SUMMARY\n")
            f.write("==========================================================\n\n")
            
            f.write(f"User ID: {self.user_id}\n\n")
            
            if self.cf_recommendations:
                f.write("1. Collaborative Filtering Recommendations:\n")
                for i, (item, score) in enumerate(list(self.cf_recommendations.items())[:10]):
                    f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
                f.write("\n")
            
            if self.pattern_recommendations:
                f.write("2. Pattern-Based Recommendations:\n")
                for i, (item, score) in enumerate(list(self.pattern_recommendations.items())[:10]):
                    f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
                f.write("\n")
            
            if self.hybrid_recommendations:
                f.write("3. Hybrid Recommendations:\n")
                for i, (item, score) in enumerate(list(self.hybrid_recommendations.items())[:10]):
                    f.write(f"   {i+1}. {item} (Score: {score:.4f})\n")
                f.write("\n")
            
            f.write("4. Integration Insights:\n")
            f.write("   - Hybrid recommendations combine collaborative filtering and pattern mining results\n")
            f.write("   - Items appearing in both recommendation sets receive a 20% boost in score\n")
            f.write("   - Collaborative filtering contributes 60% to the final score\n")
            f.write("   - Pattern mining contributes 40% to the final score\n\n")
            
            # Add user purchase history
            user_data = get_user_transactions(self.data, self.user_id)
            if len(user_data) > 0:
                f.write("5. User Purchase History:\n")
                user_items = user_data['itemDescription'].value_counts().head(10)
                for i, (item, count) in enumerate(user_items.items()):
                    f.write(f"   {i+1}. {item} ({count} purchases)\n")
        
        print(f"Results exported to 'results' directory.")

# Example usage if run directly
if __name__ == "__main__":
    try:
        # Load datasets
        train_path = os.path.join('..', '..', 'dataset', 'train.csv')
        test_path = os.path.join('..', '..', 'dataset', 'test.csv')
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Initialize and run integrated system
        system = IntegratedSystem(train_data, test_data)
        system.mine_patterns()
        system.prepare_collaborative_filtering()
        system.generate_recommendations()
        system.visualize_results()
        system.export_results()
        
    except Exception as e:
        print(f"Error: {e}") 