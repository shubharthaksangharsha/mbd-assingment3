import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data):
    """
    Preprocess the transaction data for pattern mining and recommendations
    
    Args:
        data (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle missing values
    if df.isna().any().any():
        print(f"Found {df.isna().sum().sum()} missing values in the dataset")
        # Fill missing values in itemDescription with 'unknown' or drop
        if 'itemDescription' in df.columns and df['itemDescription'].isna().any():
            print(f"Removing {df['itemDescription'].isna().sum()} records with missing item descriptions")
            df = df.dropna(subset=['itemDescription'])
    
    # Convert Date to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    except ValueError:
        # Try alternate formats if the first one fails
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except ValueError:
            print("Warning: Could not parse dates, keeping as string")
    
    # Clean item descriptions (remove leading/trailing spaces, convert to lowercase)
    if 'itemDescription' in df.columns:
        df['itemDescription'] = df['itemDescription'].astype(str).str.strip().str.lower()
        
        # Remove any problematic rows
        df = df[~df['itemDescription'].isin(['nan', 'null', ''])]
    
    return df

def create_transaction_lists(data):
    """
    Convert transaction data to lists of items purchased by each user on each date
    
    Args:
        data (pd.DataFrame): Preprocessed transaction data
        
    Returns:
        dict: Dictionary with (user_id, date) as key and list of items as value
    """
    # Group data by user_id and date, aggregating items
    transactions = {}
    for _, row in data.iterrows():
        key = (row['User_id'], row['Date'])
        if key not in transactions:
            transactions[key] = []
        
        # Only add non-empty, non-NaN items
        if pd.notna(row['itemDescription']) and row['itemDescription'] not in ('', 'nan', 'null'):
            transactions[key].append(row['itemDescription'])
    
    # Filter out empty transactions
    transactions = {k: v for k, v in transactions.items() if v}
    
    return transactions

def get_user_transactions(data, user_id):
    """
    Get all transactions for a specific user
    
    Args:
        data (pd.DataFrame): Preprocessed transaction data
        user_id (int): User ID
        
    Returns:
        pd.DataFrame: User's transaction data
    """
    return data[data['User_id'] == user_id]

def get_all_users(data):
    """
    Get list of all unique users
    
    Args:
        data (pd.DataFrame): Transaction data
        
    Returns:
        list: List of unique user IDs
    """
    return data['User_id'].unique().tolist()

def get_all_items(data):
    """
    Get list of all unique items
    
    Args:
        data (pd.DataFrame): Transaction data
        
    Returns:
        list: List of unique item descriptions
    """
    return data['itemDescription'].unique().tolist()

def create_user_item_matrix(data):
    """
    Create user-item matrix with purchase frequency
    
    Args:
        data (pd.DataFrame): Preprocessed transaction data
        
    Returns:
        pd.DataFrame: User-item matrix
    """
    # Count purchases of each item by each user
    user_item_counts = data.groupby(['User_id', 'itemDescription']).size().reset_index(name='count')
    
    # Pivot to create matrix
    user_item_matrix = user_item_counts.pivot(
        index='User_id', 
        columns='itemDescription', 
        values='count'
    ).fillna(0)
    
    return user_item_matrix

def plot_purchase_frequency(data, top_n=20):
    """
    Plot the purchase frequency of the top N items
    
    Args:
        data (pd.DataFrame): Transaction data
        top_n (int): Number of top items to display
    """
    plt.figure(figsize=(12, 8))
    
    # Count occurrences of each item
    item_counts = data['itemDescription'].value_counts().head(top_n)
    
    # Create bar plot
    sns.barplot(x=item_counts.values, y=item_counts.index)
    
    plt.title(f'Top {top_n} Most Purchased Items')
    plt.xlabel('Purchase Count')
    plt.ylabel('Item')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('visualizations/purchase_frequency.png')
    plt.close()
    
    return 'visualizations/purchase_frequency.png'

def plot_purchase_trends(data, interval='M'):
    """
    Plot purchase trends over time
    
    Args:
        data (pd.DataFrame): Transaction data
        interval (str): Time interval for grouping ('D' for daily, 'W' for weekly, 'M' for monthly)
    """
    plt.figure(figsize=(14, 8))
    
    # Use proper frequency strings to avoid FutureWarning
    freq_mapping = {
        'D': 'D',     # day
        'W': 'W-MON', # week starting Monday
        'M': 'MS'     # month start
    }
    
    freq_to_use = freq_mapping.get(interval, 'MS')
    
    # Group data by date and count transactions
    time_series = data.groupby(pd.Grouper(key='Date', freq=freq_to_use)).size()
    
    # Plot time series
    time_series.plot()
    
    interval_names = {'D': 'Daily', 'W-MON': 'Weekly', 'MS': 'Monthly'}
    title_interval = interval_names.get(freq_to_use, interval)
    
    plt.title(f'{title_interval} Purchase Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Purchases')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('visualizations/purchase_trends.png')
    plt.close()
    
    return 'visualizations/purchase_trends.png'

def add_time_features(data):
    """
    Add time-based features to the data for recency analysis
    
    Args:
        data (pd.DataFrame): Preprocessed transaction data
        
    Returns:
        pd.DataFrame: Data with added time features
    """
    df = data.copy()
    
    # Calculate days since last purchase
    max_date = df['Date'].max()
    df['days_since'] = (max_date - df['Date']).dt.days
    
    # Add recency weight (higher for more recent purchases)
    df['recency_weight'] = 1 / (df['days_since'] + 1)
    
    return df

def plot_item_co_occurrence(transaction_lists, top_n=15, output_file='visualizations/item_co_occurrence.png'):
    """
    Create a heatmap to visualize item co-occurrence in transactions
    
    Args:
        transaction_lists (list): List of transactions, where each transaction is a list of items
        top_n (int): Number of top items to include in the heatmap
        output_file (str): Path to save the output visualization
        
    Returns:
        str: Path to the saved visualization
    """
    # Flatten all transactions to get item frequencies
    all_items = []
    for transaction in transaction_lists:
        all_items.extend(transaction)
    
    # Get top N most frequent items
    item_counts = pd.Series(all_items).value_counts()
    top_items = item_counts.head(top_n).index.tolist()
    
    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(top_items), len(top_items)))
    
    # Count co-occurrences in transactions
    for transaction in transaction_lists:
        # Get items in this transaction that are in the top N
        items_in_transaction = [item for item in transaction if item in top_items]
        # Count co-occurrences
        for i, item1 in enumerate(top_items):
            for j, item2 in enumerate(top_items):
                if item1 in items_in_transaction and item2 in items_in_transaction:
                    co_occurrence[i, j] += 1
    
    # Convert to DataFrame for better visualization
    co_occurrence_df = pd.DataFrame(co_occurrence, index=top_items, columns=top_items)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence_df, annot=True, cmap='YlGnBu', fmt='.0f')
    plt.title('Item Co-occurrence Matrix (Top 15 Items)')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file)
    plt.close()
    
    return output_file 