import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import os
import sys

# Add parent directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_utils import (
    preprocess_data, 
    create_transaction_lists, 
    get_user_transactions,
    add_time_features,
    plot_item_co_occurrence
)

class PatternMiner:
    """
    Class for mining frequent patterns from transaction data
    Implements both Apriori and FP-Growth algorithms
    """
    
    def __init__(self, data):
        """
        Initialize the PatternMiner
        
        Args:
            data (pd.DataFrame): Raw transaction data
        """
        self.raw_data = data
        self.data = preprocess_data(data)
        self.transaction_lists = None
        self.encoded_data = None
        self.frequent_itemsets = None
        self.association_rules = None
        self.execution_time = None
        self.algorithm = None
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
    
    def prepare_data(self):
        """Prepare the data for pattern mining"""
        print("Preparing data for pattern mining...")
        
        # Create transaction lists
        transaction_data = create_transaction_lists(self.data)
        
        # Filter out any NaN values from transaction lists
        cleaned_transactions = []
        for transaction in transaction_data.values():
            # Remove any NaN values from the transaction
            cleaned_transaction = [item for item in transaction if isinstance(item, str) and pd.notna(item)]
            # Only add non-empty transactions
            if cleaned_transaction:
                cleaned_transactions.append(cleaned_transaction)
        
        self.transaction_lists = cleaned_transactions
        
        print(f"Preprocessing: Removed NaN values from transactions")
        print(f"Prepared {len(self.transaction_lists)} transactions")
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(self.transaction_lists).transform(self.transaction_lists)
        self.encoded_data = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"Encoded {len(self.transaction_lists)} transactions with {len(te.columns_)} unique items.")
    
    def run(self, algorithm='apriori', min_support=0.01, min_confidence=0.5, min_lift=1.0):
        """
        Run the pattern mining algorithm
        
        Args:
            algorithm (str): Mining algorithm to use ('apriori' or 'fpgrowth')
            min_support (float): Minimum support threshold
            min_confidence (float): Minimum confidence threshold
            min_lift (float): Minimum lift threshold
        """
        self.algorithm = algorithm
        if self.encoded_data is None:
            self.prepare_data()
        
        print(f"Running {algorithm.upper()} algorithm with min_support={min_support}...")
        start_time = time.time()
        
        # Run selected algorithm
        if algorithm.lower() == 'apriori':
            self.frequent_itemsets = apriori(
                self.encoded_data, 
                min_support=min_support, 
                use_colnames=True,
                verbose=1
            )
        elif algorithm.lower() == 'fpgrowth':
            self.frequent_itemsets = fpgrowth(
                self.encoded_data,
                min_support=min_support,
                use_colnames=True,
                verbose=1
            )
        else:
            raise ValueError("Algorithm must be either 'apriori' or 'fpgrowth'")
        
        execution_time = time.time() - start_time
        self.execution_time = execution_time
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.association_rules = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            self.association_rules = self.association_rules[self.association_rules['lift'] >= min_lift]
            
            print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
            print(f"Generated {len(self.association_rules)} association rules")
        else:
            print("No frequent itemsets found with the given minimum support.")
            self.association_rules = pd.DataFrame()
        
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return self.frequent_itemsets, self.association_rules
    
    def score_patterns(self, user_id=None):
        """
        Score patterns by importance/quality
        If user_id is provided, score patterns relevant to that user
        
        Args:
            user_id (int, optional): User ID to score patterns for
            
        Returns:
            pd.DataFrame: Scored patterns
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("No frequent itemsets available. Run the algorithm first.")
            return pd.DataFrame()
        
        # Add pattern length and a combined score
        scored_patterns = self.frequent_itemsets.copy()
        scored_patterns['length'] = scored_patterns['itemsets'].apply(lambda x: len(x))
        
        # Combined score: support * length (favors longer patterns with good support)
        scored_patterns['score'] = scored_patterns['support'] * scored_patterns['length'] * 2.0
        
        # If user_id is provided, filter and score based on user's purchase history
        if user_id is not None:
            user_data = get_user_transactions(self.data, user_id)
            
            if len(user_data) > 0:
                # Get user's purchased items
                user_items = set(user_data['itemDescription'].unique())
                
                # Score patterns based on overlap with user's purchases
                def calculate_user_relevance(itemset):
                    itemset_items = set(itemset)
                    overlap = len(itemset_items.intersection(user_items))
                    if overlap > 0:
                        # Give higher weight to overlap - square it to amplify effect
                        return (overlap / len(itemset_items)) ** 2
                    return 0
                
                scored_patterns['user_relevance'] = scored_patterns['itemsets'].apply(calculate_user_relevance)
                
                # Apply time-based weighting if there are multiple purchases
                if len(user_data) > 1:
                    user_data_with_time = add_time_features(user_data)
                    
                    # Calculate average recency weight for each item
                    item_recency = user_data_with_time.groupby('itemDescription')['recency_weight'].mean()
                    
                    # Score patterns based on recency of items
                    def calculate_recency_score(itemset):
                        itemset_items = set(itemset)
                        recency_scores = [item_recency.get(item, 0) for item in itemset_items if item in item_recency.index]
                        if recency_scores:
                            return sum(recency_scores) / len(recency_scores)
                        return 0
                    
                    scored_patterns['recency_score'] = scored_patterns['itemsets'].apply(calculate_recency_score)
                    
                    # Combined score including user relevance and recency
                    scored_patterns['score'] = scored_patterns['score'] * (1 + 3 * scored_patterns['user_relevance']) * (1 + scored_patterns['recency_score'])
                else:
                    # If only one purchase, just use user relevance with higher weight
                    scored_patterns['score'] = scored_patterns['score'] * (1 + 3 * scored_patterns['user_relevance'])
            
        # Sort by score in descending order
        scored_patterns = scored_patterns.sort_values('score', ascending=False)
        
        return scored_patterns
    
    def get_patterns_for_user(self, user_id, top_n=10):
        """
        Get top patterns relevant to a specific user
        
        Args:
            user_id (int): User ID
            top_n (int): Number of top patterns to return
            
        Returns:
            pd.DataFrame: Top patterns for the user
        """
        scored_patterns = self.score_patterns(user_id)
        
        if len(scored_patterns) == 0:
            return pd.DataFrame()
            
        return scored_patterns.head(top_n)
    
    def visualize_results(self):
        """Generate visualizations of the mining results"""
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("No frequent itemsets available to visualize.")
            return
        
        # 1. Plot support distribution of frequent itemsets
        plt.figure(figsize=(10, 6))
        plt.hist(self.frequent_itemsets['support'], bins=20)
        plt.title(f'Support Distribution of Frequent Itemsets ({self.algorithm.upper()})')
        plt.xlabel('Support')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/pattern_support_distribution.png')
        plt.close()
        
        # 2. Plot itemset length distribution
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(lambda x: len(x))
        plt.figure(figsize=(10, 6))
        sns.countplot(x='length', data=self.frequent_itemsets)
        plt.title(f'Length Distribution of Frequent Itemsets ({self.algorithm.upper()})')
        plt.xlabel('Itemset Length')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/pattern_itemset_length_distribution.png')
        plt.close()
        
        # 3. Plot top items by frequency in itemsets
        if len(self.frequent_itemsets) > 0:
            # Extract all items from frequent itemsets
            all_items = []
            for itemset in self.frequent_itemsets['itemsets']:
                all_items.extend(list(itemset))
            
            # Count item occurrences
            item_counts = pd.Series(all_items).value_counts().head(15)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=item_counts.values, y=item_counts.index)
            plt.title(f'Top 15 Most Frequent Items in Itemsets ({self.algorithm.upper()})')
            plt.xlabel('Frequency')
            plt.ylabel('Item')
            plt.tight_layout()
            plt.savefig('visualizations/pattern_top_frequent_items.png')
            plt.close()
        
        # 4. If association rules exist, visualize them
        if self.association_rules is not None and len(self.association_rules) > 0:
            # Plot scatter plot of lift vs confidence
            plt.figure(figsize=(12, 10))
            
            # Enhance scatter plot with more visual info
            scatter = plt.scatter(
                self.association_rules['confidence'],
                self.association_rules['lift'],
                alpha=0.7,
                s=self.association_rules['support']*1000,  # Larger point size
                c=self.association_rules['lift'],  # Color by lift
                cmap='viridis',
                edgecolors='black',
                linewidths=1
            )
            
            # Add colorbar to show lift values
            plt.colorbar(scatter, label='Lift Value')
            
            # Add rule count annotation
            plt.annotate(f'Total Rules: {len(self.association_rules)}', 
                         xy=(0.05, 0.95), 
                         xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Improve labels and styling
            plt.title('Association Rules: Lift vs Confidence', fontsize=16)
            plt.xlabel('Confidence', fontsize=14)
            plt.ylabel('Lift', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Ensure we can see all points clearly
            plt.tight_layout()
            plt.savefig('visualizations/pattern_rules_lift_confidence.png', dpi=150)
            plt.close()
            
            # Plot top 10 rules by lift
            top_rules = self.association_rules.sort_values('lift', ascending=False).head(10)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_rules['lift'], y=top_rules.index)
            plt.title('Top 10 Association Rules by Lift')
            plt.xlabel('Lift')
            plt.ylabel('Rule Index')
            plt.tight_layout()
            plt.savefig('visualizations/pattern_top_rules_by_lift.png')
            plt.close()
            
            # 5. ADDED: Visualize rule network (if networkx is available)
            try:
                import networkx as nx
                
                # Create a directed graph
                G = nx.DiGraph()
                
                # Add edges for top 20 rules by lift
                top_rules_network = self.association_rules.sort_values('lift', ascending=False).head(20)
                
                # Add nodes and edges
                for _, rule in top_rules_network.iterrows():
                    # Convert frozensets to strings for better node labels
                    antecedent = ', '.join(list(rule['antecedents']))
                    consequent = ', '.join(list(rule['consequents']))
                    
                    # Add the edge with attributes
                    G.add_edge(
                        antecedent, 
                        consequent, 
                        weight=rule['lift'], 
                        confidence=rule['confidence'],
                        support=rule['support']
                    )
                
                # Draw the graph
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G, seed=42)  # positions for all nodes
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
                
                # Draw edges with width based on lift
                edges = G.edges()
                weights = [G[u][v]['weight'] * 0.5 for u, v in edges]
                nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrows=True, arrowsize=20)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
                
                plt.axis('off')
                plt.title('Association Rules Network (Top 20 by Lift)')
                plt.tight_layout()
                plt.savefig('visualizations/pattern_rules_network.png')
                plt.close()
                print("Generated association rules network visualization")
            except ImportError:
                print("Networkx not installed, skipping rule network visualization")
        
        # 6. NEW: Create item co-occurrence heatmap
        if self.transaction_lists is not None and len(self.transaction_lists) > 0:
            co_occurrence_path = plot_item_co_occurrence(
                self.transaction_lists, 
                top_n=15,
                output_file='visualizations/pattern_item_co_occurrence.png'
            )
            print(f"Generated item co-occurrence visualization: {co_occurrence_path}")
        
        print("Visualizations saved to 'visualizations' directory.")
    
    def export_results(self, output_dir='results'):
        """Export the results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            # Convert itemsets to string for CSV export
            export_itemsets = self.frequent_itemsets.copy()
            export_itemsets['itemsets_str'] = export_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
            
            # Export frequent itemsets
            export_itemsets.to_csv(f'{output_dir}/frequent_itemsets_{self.algorithm}.csv', index=False)
            
            if self.association_rules is not None and len(self.association_rules) > 0:
                # Convert rule antecedents and consequents to string
                export_rules = self.association_rules.copy()
                export_rules['antecedents_str'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                export_rules['consequents_str'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                # Export association rules
                export_rules.to_csv(f'{output_dir}/association_rules_{self.algorithm}.csv', index=False)
            
            print(f"Results exported to '{output_dir}' directory.")
        else:
            print("No results to export.")
            
# Example usage if run directly
if __name__ == "__main__":
    try:
        # Load dataset
        train_path = os.path.join('..', '..', 'dataset', 'train.csv')
        train_data = pd.read_csv(train_path)
        
        # Initialize and run pattern mining
        miner = PatternMiner(train_data)
        miner.run(algorithm='apriori', min_support=0.01)
        miner.visualize_results()
        miner.export_results()
        
    except Exception as e:
        print(f"Error: {e}") 