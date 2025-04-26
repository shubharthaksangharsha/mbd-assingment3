import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import time
import os
import sys
import random
from collections import defaultdict

# Add parent directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_utils import (
    preprocess_data,
    create_user_item_matrix,
    get_user_transactions,
    get_all_users,
    get_all_items,
    add_time_features
)

class CollaborativeFilter:
    """
    Class for collaborative filtering-based recommendation system
    Implements both memory-based (user-user, item-item) and model-based (SVD) approaches
    """
    
    def __init__(self, data):
        """
        Initialize the CollaborativeFilter
        
        Args:
            data (pd.DataFrame): Raw transaction data
        """
        self.raw_data = data
        self.data = preprocess_data(data)
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.svd_matrix = None
        self.recommendations = None
        self.user_id = None
        self.method = None
        self.execution_time = None
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
    
    def prepare_data(self):
        """Prepare the data for collaborative filtering"""
        print("Preparing data for collaborative filtering...")
        
        # Create user-item matrix
        self.user_item_matrix = create_user_item_matrix(self.data)
        
        # Fill NaN values with 0 (no purchase)
        self.user_item_matrix = self.user_item_matrix.fillna(0)
        
        print(f"Created user-item matrix of shape: {self.user_item_matrix.shape}")
        print(f"Number of users: {self.user_item_matrix.shape[0]}")
        print(f"Number of items: {self.user_item_matrix.shape[1]}")
    
    def compute_similarity(self, mode='user'):
        """
        Compute similarity matrix for users or items
        
        Args:
            mode (str): Mode for similarity computation ('user' or 'item')
        """
        if self.user_item_matrix is None:
            self.prepare_data()
            
        print(f"Computing {mode} similarity matrix...")
        start_time = time.time()
        
        if mode == 'user':
            # Compute user similarity matrix
            self.user_similarity = cosine_similarity(self.user_item_matrix)
            # Convert to DataFrame for easier indexing
            self.user_similarity = pd.DataFrame(
                self.user_similarity,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
        elif mode == 'item':
            # Compute item similarity matrix
            self.item_similarity = cosine_similarity(self.user_item_matrix.T)
            # Convert to DataFrame for easier indexing
            self.item_similarity = pd.DataFrame(
                self.item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        else:
            raise ValueError("Mode must be either 'user' or 'item'")
            
        execution_time = time.time() - start_time
        print(f"Similarity computation completed in {execution_time:.2f} seconds")
    
    def train_svd_model(self, n_components=50):
        """
        Train a model-based SVD model for recommendations
        
        Args:
            n_components (int): Number of latent factors
        """
        if self.user_item_matrix is None:
            self.prepare_data()
            
        print(f"Training SVD model with {n_components} components...")
        start_time = time.time()
        
        # Initialize SVD model
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit SVD model
        self.svd_matrix = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Get explained variance
        explained_variance = self.svd_model.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_variance:.4f}")
        
        execution_time = time.time() - start_time
        print(f"SVD training completed in {execution_time:.2f} seconds")
    
    def generate_recommendations(self, user_id=None, method='user-based', n_recommendations=5, use_time_weight=True):
        """
        Generate recommendations for a user using the specified method
        
        Args:
            user_id (int, optional): User ID to generate recommendations for
            method (str): Recommendation method ('user-based', 'item-based', or 'svd')
            n_recommendations (int): Number of recommendations to generate
            use_time_weight (bool): Whether to use time-based weighting
            
        Returns:
            pd.DataFrame: Top N recommendations for the user
        """
        if self.user_item_matrix is None:
            self.prepare_data()
            
        # If user_id is not provided, select a random user
        if user_id is None:
            user_id = random.choice(self.user_item_matrix.index.tolist())
            print(f"No user ID provided. Selected random user: {user_id}")
        
        # Check if user exists in the dataset
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset. Selecting random user.")
            user_id = random.choice(self.user_item_matrix.index.tolist())
            print(f"Selected random user: {user_id}")
        
        # Store the user ID for later use
        self.user_id = user_id
        self.method = method
        
        # Get user's purchase history
        user_data = get_user_transactions(self.data, user_id)
        user_items = set(user_data['itemDescription'].unique())
        
        print(f"User {user_id} has purchased {len(user_items)} unique items")
        
        # Generate recommendations using the specified method
        start_time = time.time()
        
        if method == 'user-based':
            # Ensure user similarity matrix is computed
            if self.user_similarity is None:
                self.compute_similarity(mode='user')
            
            recommendations = self._user_based_recommendations(user_id, n_recommendations, user_items, use_time_weight)
        
        elif method == 'item-based':
            # Ensure item similarity matrix is computed
            if self.item_similarity is None:
                self.compute_similarity(mode='item')
            
            recommendations = self._item_based_recommendations(user_id, n_recommendations, user_items, use_time_weight)
        
        elif method == 'svd':
            # Ensure SVD model is trained
            if self.svd_model is None:
                self.train_svd_model()
            
            recommendations = self._svd_recommendations(user_id, n_recommendations, user_items)
        
        else:
            raise ValueError("Method must be 'user-based', 'item-based', or 'svd'")
        
        execution_time = time.time() - start_time
        self.execution_time = execution_time
        
        # Store recommendations for later use
        self.recommendations = recommendations
        
        print(f"Generated {len(recommendations)} recommendations in {execution_time:.2f} seconds")
        print("\nTop recommendations:")
        for i, (item, score) in enumerate(recommendations.items()):
            print(f"{i+1}. {item} (Score: {score:.4f})")
        
        return recommendations
    
    def _user_based_recommendations(self, user_id, n_recommendations, user_items, use_time_weight=True):
        """
        Generate user-based recommendations
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            user_items (set): Set of items the user has already purchased
            use_time_weight (bool): Whether to use time-based weighting
            
        Returns:
            dict: Recommended items with scores
        """
        # Get similarity scores for this user
        user_similarities = self.user_similarity.loc[user_id]
        
        # Dictionary to store recommendation scores
        recommendations = defaultdict(float)
        
        # Get similar users
        for other_user, similarity in user_similarities.items():
            # Skip the user themselves and users with zero similarity
            if other_user == user_id or similarity <= 0:
                continue
            
            # Get items purchased by the other user
            other_user_purchases = self.user_item_matrix.loc[other_user]
            other_user_purchases = other_user_purchases[other_user_purchases > 0]
            
            # For each item the other user has purchased
            for item, count in other_user_purchases.items():
                # Skip items the user has already purchased
                if item in user_items:
                    continue
                
                # Add weighted score to recommendations
                recommendations[item] += similarity * count
        
        # Apply time-based weighting if enabled
        if use_time_weight and len(user_items) > 0:
            recommendations = self._apply_time_weighting(recommendations, user_id)
        
        # Sort recommendations by score in descending order
        recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
        
        return recommendations
    
    def _item_based_recommendations(self, user_id, n_recommendations, user_items, use_time_weight=True):
        """
        Generate item-based recommendations
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            user_items (set): Set of items the user has already purchased
            use_time_weight (bool): Whether to use time-based weighting
            
        Returns:
            dict: Recommended items with scores
        """
        # Dictionary to store recommendation scores
        recommendations = defaultdict(float)
        
        # If the user has no purchase history, return random recommendations
        if len(user_items) == 0:
            print("User has no purchase history. Generating random recommendations.")
            all_items = set(self.user_item_matrix.columns)
            random_items = random.sample(all_items, min(n_recommendations, len(all_items)))
            return {item: 1.0 for item in random_items}
        
        # For each item the user has purchased
        for user_item in user_items:
            # Skip if item not in the similarity matrix (shouldn't happen, but just in case)
            if user_item not in self.item_similarity.index:
                continue
            
            # Get similarity scores for this item
            item_similarities = self.item_similarity.loc[user_item]
            
            # For each other item
            for other_item, similarity in item_similarities.items():
                # Skip the item itself and items with zero similarity
                if other_item == user_item or similarity <= 0:
                    continue
                
                # Skip items the user has already purchased
                if other_item in user_items:
                    continue
                
                # Add weighted score to recommendations
                recommendations[other_item] += similarity
        
        # Apply time-based weighting if enabled
        if use_time_weight:
            recommendations = self._apply_time_weighting(recommendations, user_id)
        
        # Sort recommendations by score in descending order
        recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
        
        return recommendations
    
    def _svd_recommendations(self, user_id, n_recommendations, user_items):
        """
        Generate SVD-based recommendations
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            user_items (set): Set of items the user has already purchased
            
        Returns:
            dict: Recommended items with scores
        """
        # Get the index of the user in the user-item matrix
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Get the user's vector in the latent space
        user_vector = self.svd_matrix[user_idx]
        
        # Get the item vectors in the latent space
        item_vectors = self.svd_model.components_.T
        
        # Calculate the predicted ratings for all items
        predicted_ratings = np.dot(user_vector, self.svd_model.components_)
        
        # Create a dictionary of item to predicted rating
        item_to_rating = dict(zip(self.user_item_matrix.columns, predicted_ratings))
        
        # Remove items the user has already purchased
        for item in user_items:
            if item in item_to_rating:
                del item_to_rating[item]
        
        # Sort by predicted rating in descending order
        recommendations = dict(sorted(item_to_rating.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
        
        return recommendations
    
    def _apply_time_weighting(self, recommendations, user_id):
        """
        Apply time-based weighting to recommendations
        
        Args:
            recommendations (dict): Dictionary of recommended items with scores
            user_id (int): User ID
            
        Returns:
            dict: Recommendations with time-based weighting applied
        """
        # Get user's transactions with time features
        user_data = get_user_transactions(self.data, user_id)
        
        if len(user_data) <= 1:
            # No time weighting if user has only one purchase or no purchases
            return recommendations
        
        # Add time features
        user_data_with_time = add_time_features(user_data)
        
        # Calculate average recency weight for each item
        item_recency = user_data_with_time.groupby('itemDescription')['recency_weight'].mean()
        
        # Calculate average recency weight across all user items
        avg_recency = item_recency.mean()
        
        # Apply recency weight to recommendations
        weighted_recommendations = {}
        for item, score in recommendations.items():
            # Add a recency boost (more weight to items similar to recently purchased items)
            weighted_recommendations[item] = score * (1 + avg_recency)
        
        return weighted_recommendations
    
    def evaluate_recommendations(self, test_data=None, n_users=100, method='user-based'):
        """
        Evaluate recommendation quality using test data
        
        Args:
            test_data (pd.DataFrame, optional): Test data for evaluation
            n_users (int): Number of users to evaluate
            method (str): Recommendation method
            
        Returns:
            dict: Evaluation metrics
        """
        if test_data is None:
            print("Warning: No test data provided. Using training data for evaluation.")
            # Split the data temporally if no test data is provided
            max_date = self.data['Date'].max()
            min_date = self.data['Date'].min()
            midpoint = min_date + (max_date - min_date) / 2
            
            train_data = self.data[self.data['Date'] <= midpoint]
            test_data = self.data[self.data['Date'] > midpoint]
        else:
            train_data = self.data
            
            # Clone the test data to avoid modifying the original
            test_data_clone = test_data.copy()
            
            # Ensure we have the User_id column before preprocessing
            if 'User_id' not in test_data_clone.columns:
                # Try to find variant spellings
                user_id_variants = ['user_id', 'userId', 'userid', 'user id', 'id']
                for variant in user_id_variants:
                    if variant in test_data_clone.columns:
                        test_data_clone.rename(columns={variant: 'User_id'}, inplace=True)
                        break
            
            # Save the User_id column before preprocessing
            user_ids = None
            if 'User_id' in test_data_clone.columns:
                # Create a mapping of original row indices to User_ids
                user_ids = {i: uid for i, uid in enumerate(test_data_clone['User_id'])}
            
            # Preprocess the test data
            test_data = preprocess_data(test_data_clone)
            
            # Re-add the User_id column if it was lost during preprocessing
            if 'User_id' not in test_data.columns and user_ids is not None:
                # Create a new User_id column based on the original row indices
                test_data['User_id'] = test_data.index.map(lambda i: user_ids.get(i, None))
                
                # Drop rows that couldn't be mapped back to a User_id
                test_data = test_data.dropna(subset=['User_id'])
            
            if 'User_id' not in test_data.columns:
                print("Error: Unable to preserve or reconstruct the User_id column in test data.")
                return {
                    'hit_rate': 0,
                    'average_precision': 0,
                    'coverage': 0,
                    'diversity': 0
                }
        
        # Get users that exist in both train and test
        train_users = set(train_data['User_id'].unique())
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
        
        print(f"Evaluating recommendations for {n_users} users using {method} method...")
        
        # Initialize evaluation metrics
        metrics = {
            'hit_rate': 0,
            'average_precision': 0,
            'coverage': set(),
            'diversity': 0
        }
        
        # Initialize a new recommender with only the training data
        recommender = CollaborativeFilter(train_data)
        
        # Set of all recommended items (for coverage calculation)
        all_recommended_items = set()
        
        # Dictionary to count recommendation co-occurrences (for diversity calculation)
        item_co_occurrences = defaultdict(int)
        
        # For each user
        for user_id in eval_users:
            # Get user's actual purchases in test data
            user_test_data = test_data[test_data['User_id'] == user_id]
            actual_items = set(user_test_data['itemDescription'].unique())
            
            # If user has no purchases in test data, skip
            if len(actual_items) == 0:
                continue
            
            # Generate recommendations for the user
            recommendations = recommender.generate_recommendations(
                user_id=user_id,
                method=method,
                n_recommendations=10,
                use_time_weight=True
            )
            
            recommended_items = list(recommendations.keys())
            
            # Update all recommended items set
            all_recommended_items.update(recommended_items)
            
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
        
        # Calculate average metrics
        metrics['hit_rate'] /= n_users
        metrics['average_precision'] /= n_users
        
        # Calculate coverage (proportion of all items that were recommended)
        all_items = set(train_data['itemDescription'].unique())
        metrics['coverage'] = len(all_recommended_items) / len(all_items) if len(all_items) > 0 else 0
        
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
    
    def compare_methods(self, user_id=None, n_recommendations=5, test_data=None):
        """
        Compare different recommendation methods
        
        Args:
            user_id (int, optional): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to generate
            test_data (pd.DataFrame, optional): Test data for evaluation
            
        Returns:
            dict: Comparison results
        """
        methods = ['user-based', 'item-based', 'svd']
        
        # If user_id is not provided, select a random user
        if user_id is None:
            if self.user_item_matrix is None:
                self.prepare_data()
            user_id = random.choice(self.user_item_matrix.index.tolist())
            print(f"No user ID provided. Selected random user: {user_id}")
        
        print(f"Comparing recommendation methods for user {user_id}...")
        
        # Generate recommendations using each method
        recommendations = {}
        execution_times = {}
        
        for method in methods:
            print(f"\nGenerating recommendations using {method} method...")
            start_time = time.time()
            
            recs = self.generate_recommendations(
                user_id=user_id,
                method=method,
                n_recommendations=n_recommendations,
                use_time_weight=True
            )
            
            execution_time = time.time() - start_time
            
            recommendations[method] = recs
            execution_times[method] = execution_time
            
            print(f"{method} method completed in {execution_time:.2f} seconds")
        
        # If test data is provided, evaluate each method
        if test_data is not None:
            evaluation_metrics = {}
            
            for method in methods:
                metrics = self.evaluate_recommendations(
                    test_data=test_data,
                    n_users=100,
                    method=method
                )
                
                evaluation_metrics[method] = metrics
        
        # Combine results
        comparison_results = {
            'user_id': user_id,
            'recommendations': recommendations,
            'execution_times': execution_times
        }
        
        if test_data is not None:
            comparison_results['evaluation_metrics'] = evaluation_metrics
        
        return comparison_results
    
    def visualize_results(self):
        """Generate visualizations of the recommendation results"""
        if self.recommendations is None or self.user_id is None:
            print("No recommendations available to visualize. Generate recommendations first.")
            return
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Plot top recommendations with scores
        plt.figure(figsize=(12, 8))
        
        items = list(self.recommendations.keys())
        scores = list(self.recommendations.values())
        
        plt.barh(items, scores)
        plt.xlabel('Recommendation Score')
        plt.ylabel('Item')
        plt.title(f'Top Recommendations for User {self.user_id} ({self.method})')
        plt.tight_layout()
        plt.savefig('visualizations/top_recommendations.png')
        plt.close()
        
        # 2. Plot user similarity matrix (if available)
        if self.user_similarity is not None and self.user_id in self.user_similarity.index:
            plt.figure(figsize=(12, 8))
            
            # Get top 20 most similar users
            user_similarities = self.user_similarity.loc[self.user_id].sort_values(ascending=False).head(21)
            user_similarities = user_similarities.drop(self.user_id)  # Remove self-similarity
            
            plt.barh(user_similarities.index.astype(str), user_similarities.values)
            plt.xlabel('Similarity Score')
            plt.ylabel('User ID')
            plt.title(f'Top 20 Most Similar Users to User {self.user_id}')
            plt.tight_layout()
            plt.savefig('visualizations/similar_users.png')
            plt.close()
        
        # 3. Plot item similarity matrix (if available)
        if self.item_similarity is not None and self.recommendations:
            plt.figure(figsize=(14, 10))
            
            # Get top 5 recommended items
            top_items = list(self.recommendations.keys())[:5]
            
            # Create a subset of the item similarity matrix for visualization
            item_sim_subset = self.item_similarity.loc[top_items, top_items]
            
            # Plot heatmap
            sns.heatmap(item_sim_subset, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Item Similarity Matrix for Top Recommended Items')
            plt.tight_layout()
            plt.savefig('visualizations/item_similarity.png')
            plt.close()
        
        # 4. Plot user purchase history vs recommendations
        user_data = get_user_transactions(self.data, self.user_id)
        
        if len(user_data) > 0:
            user_items = user_data['itemDescription'].value_counts().head(10)
            
            plt.figure(figsize=(14, 8))
            
            plt.subplot(1, 2, 1)
            plt.barh(user_items.index, user_items.values)
            plt.xlabel('Purchase Count')
            plt.ylabel('Item')
            plt.title(f'Top 10 Purchased Items by User {self.user_id}')
            
            plt.subplot(1, 2, 2)
            items = list(self.recommendations.keys())[:10]
            scores = list(self.recommendations.values())[:10]
            plt.barh(items, scores)
            plt.xlabel('Recommendation Score')
            plt.ylabel('Item')
            plt.title(f'Top 10 Recommended Items for User {self.user_id}')
            
            plt.tight_layout()
            plt.savefig('visualizations/purchase_vs_recommendations.png')
            plt.close()
        
        print("Visualizations saved to 'visualizations' directory.")
    
    def export_results(self, output_dir='results'):
        """Export the results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.recommendations is None or self.user_id is None:
            print("No recommendations available to export. Generate recommendations first.")
            return
        
        # Export recommendations
        recommendations_df = pd.DataFrame({
            'item': list(self.recommendations.keys()),
            'score': list(self.recommendations.values())
        })
        
        recommendations_df.to_csv(f'{output_dir}/recommendations_{self.method}_{self.user_id}.csv', index=False)
        
        # Export user-item matrix statistics
        if self.user_item_matrix is not None:
            matrix_stats = pd.DataFrame({
                'statistic': ['num_users', 'num_items', 'sparsity'],
                'value': [
                    self.user_item_matrix.shape[0],
                    self.user_item_matrix.shape[1],
                    1 - (self.user_item_matrix > 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
                ]
            })
            
            matrix_stats.to_csv(f'{output_dir}/user_item_matrix_stats.csv', index=False)
        
        print(f"Results exported to '{output_dir}' directory.")

# Example usage if run directly
if __name__ == "__main__":
    try:
        # Load dataset
        train_path = "/home/shubharthak/Desktop/mbd-assingment3/dataset/train.csv"
        train_data = pd.read_csv(train_path)
        
        # Initialize and run collaborative filtering
        cf = CollaborativeFilter(train_data)
        cf.generate_recommendations(method='user-based')
        cf.visualize_results()
        cf.export_results()
        
    except Exception as e:
        print(f"Error: {e}") 