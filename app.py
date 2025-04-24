import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import base64
import uuid
import re
import random
from collections import Counter
from io import StringIO
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from matplotlib_venn import venn2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import project components
from src.task_a.pattern_mining import PatternMiner
from src.task_b.collaborative_filtering import CollaborativeFilter
from src.task_c.integration import IntegratedSystem

# Set page configuration
st.set_page_config(
    page_title="Grocery Store Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for pastel colors
st.markdown("""
<style>
    .main {
        background-color: #F8F9FA;
    }
    .stApp {
        background: linear-gradient(to bottom right, #f0f7ff, #e8f4f8);
    }
    .sub-header {
        color: #2e7ad1;
        padding: 10px 0;
        border-bottom: 2px solid #4b91e2;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText {
        color: #1a3c61 !important;
    }
    .stButton>button {
        background-color: #4b91e2;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e7ad1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e1f0ff;
        border-left: 5px solid #4b91e2;
        padding: 10px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stMetric {
        background-color: #e1f0ff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e1f0ff;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #2e7ad1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4b91e2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Add style for the info-box in Task C
st.markdown("""
<style>
    .info-box {
        background-color: rgba(225, 240, 255, 0.9);
        border-left: 5px solid #4b91e2;
        padding: 15px;
        border-radius: 0 5px 5px 0;
        margin: 15px 0;
        color: #1a3c61;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Task C specific styles */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Improve tab visibility in Task C */
    [data-testid="stTabs"] > div:first-child {
        background-color: rgba(225, 240, 255, 0.5);
        border-radius: 8px 8px 0 0;
    }
    
    /* Ensure high contrast for all text elements */
    div.stMarkdown p, div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3, 
    div.stMarkdown h4, div.stMarkdown li, div.stMarkdown a {
        color: #1a3c61 !important;
    }
    
    /* Fix for info alerts */
    [data-baseweb="notification"] {
        background-color: rgba(225, 240, 255, 0.9) !important;
    }
    
    /* Make success messages more visible */
    [data-baseweb="notification"][kind="positive"] {
        background-color: rgba(227, 250, 239, 0.9) !important;
    }
    
    /* User selection section styling */
    div[data-testid="stRadio"] > div {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    div[data-testid="stRadio"] label {
        color: #1a3c61 !important;
        font-weight: 500;
    }
    
    div[data-testid="stRadio"] input {
        accent-color: #4b91e2;
    }
    
    /* Additional button styling for better visibility */
    .stButton > button {
        background-color: #4b91e2 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .stButton > button:hover {
        background-color: #2e7ad1 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Metrics section styling */
    [data-testid="stMetric"] {
        background-color: rgba(225, 240, 255, 0.8);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetric"] label {
        color: #1a3c61 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetric"] div {
        color: #2e7ad1 !important;
    }
    
    /* System setup section */
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #2e7ad1 !important;
        font-weight: 600;
        border-bottom: 1px solid #4b91e2;
        padding-bottom: 5px;
    }
    
    /* Fix selectbox text visibility */
    div[data-baseweb="select"] span {
        color: #1a3c61 !important;
    }
    
    /* Overall section headings */
    h3 {
        color: #2e7ad1 !important;
        padding-top: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #4b91e2;
        margin-bottom: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🛒 Grocery Store Analytics Dashboard")
st.markdown("### Insights from Grocery Store Purchase Data")
st.markdown("Explore user behavior, product trends, and test various recommendation methods to understand shopping patterns.")

# Function to safely convert values to string for search
def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)

# Function to create download button for any content
def create_download_button(object_to_download, download_filename, button_text):
    """
    Generates a download button for the provided data
    
    Parameters:
    -----------
    object_to_download : DataFrame or str
        The object to be downloaded (DataFrame or string)
    download_filename : str
        The name of the file to be downloaded
    button_text : str
        The text to display on the download button
    
    Returns:
    --------
    None
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Create a unique ID for this button
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub(r'\d+', '', button_uuid)
    
    b64 = base64.b64encode(object_to_download.encode()).decode()
    dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:text/csv;base64,{b64}" target="_blank">{button_text}</a>'
    
    button_css = f"""
        <style>
            #{button_id} {{
                background-color: #a779e3;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 6px;
                border: none;
                transition-duration: 0.4s;
            }}
            #{button_id}:hover {{
                background-color: #7b50a6;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            }}
        </style>
    """
    
    st.markdown(button_css + dl_link, unsafe_allow_html=True)

# Create directory structure if it doesn't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Check if dataset exists
def check_dataset():
    train_path = os.path.join('dataset', 'train.csv')
    test_path = os.path.join('dataset', 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        st.error(f"Dataset files not found! Please ensure train.csv and test.csv are in the dataset folder.")
        return False
    return True

# Load the dataset
@st.cache_data
def load_data():
    train_path = os.path.join('dataset', 'train.csv')
    test_path = os.path.join('dataset', 'test.csv')
    
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Standardize User_id column names
        user_id_variants = ['User_id', 'user_id', 'userId', 'userid', 'user id', 'id']
        
        # For train data
        if 'User_id' not in train_data.columns:
            for variant in user_id_variants:
                if variant in train_data.columns:
                    train_data.rename(columns={variant: 'User_id'}, inplace=True)
                    break
                    
        # For test data
        if 'User_id' not in test_data.columns:
            for variant in user_id_variants:
                if variant in test_data.columns:
                    test_data.rename(columns={variant: 'User_id'}, inplace=True)
                    break
        
        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a task",
    ["Home", "Task A: Pattern Mining", "Task B: Collaborative Filtering", 
     "Task C: Integrated System", "Task D: Interactive Analysis"]
)

# Initialize session state for storing data between interactions
if 'train_data' not in st.session_state:
    st.session_state.train_data = None

if 'test_data' not in st.session_state:
    st.session_state.test_data = None

if 'pattern_miner' not in st.session_state:
    st.session_state.pattern_miner = None

if 'collaborative_filter' not in st.session_state:
    st.session_state.collaborative_filter = None

if 'integrated_system' not in st.session_state:
    st.session_state.integrated_system = None

# Load data only if the dataset exists
if check_dataset():
    if st.session_state.train_data is None or st.session_state.test_data is None:
        with st.spinner("Loading dataset..."):
            st.session_state.train_data, st.session_state.test_data = load_data()
            st.success(f"Successfully loaded training data: {len(st.session_state.train_data)} records")
            st.success(f"Successfully loaded test data: {len(st.session_state.test_data)} records")

# Main content based on selected mode
if app_mode == "Home":
    # Display home page with overview and dataset summary
    st.markdown("<h2 class='sub-header'>Welcome to the Grocery Store Recommendation System</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### About this Application")
        st.markdown("""
        This application demonstrates data mining and recommendation techniques for a grocery store dataset:
        
        - **Pattern Mining**: Discover frequent purchase patterns and association rules
        - **Collaborative Filtering**: Generate personalized recommendations based on user similarity
        - **Integrated System**: Combine pattern mining and collaborative filtering
        - **Interactive Analysis**: Explore shopping patterns and compare recommendations
        """)
    
    with col2:
        st.markdown("### Dataset Overview")
        if st.session_state.train_data is not None:
            st.dataframe(st.session_state.train_data.head(5))
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Users", f"{st.session_state.train_data['User_id'].nunique():,}")
            with col_b:
                st.metric("Products", f"{st.session_state.train_data['itemDescription'].nunique():,}")
            with col_c:
                st.metric("Transactions", f"{len(st.session_state.train_data):,}")
                
            st.markdown("#### Data Summary")
            st.write(st.session_state.train_data.describe())

# Task specific code will be added in subsequent steps

# Task A: Pattern Mining
elif app_mode == "Task A: Pattern Mining":
    st.markdown("<h2 class='sub-header'>Task A: Pattern Mining</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Pattern mining discovers frequent purchase patterns and association rules from transaction data.
    This helps identify which products are often bought together and can be used for product placement,
    promotions, and initial recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize pattern miner if not already done
    if st.session_state.pattern_miner is None and st.session_state.train_data is not None:
        st.session_state.pattern_miner = PatternMiner(st.session_state.train_data)
    
    # Configuration options
    st.markdown("### Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.radio(
            "Select Mining Algorithm",
            ["Apriori", "FP-Growth"],
            index=1,
            help="Apriori is easier to understand but slower. FP-Growth is faster and better for large datasets."
        )
        
        algorithm_code = 'apriori' if algorithm == "Apriori" else 'fpgrowth'
        
    with col2:
        min_support = st.slider(
            "Minimum Support Threshold",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="The minimum frequency of an itemset to be considered 'frequent'. Lower values find more patterns but take longer."
        )
        
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="The minimum confidence for association rules. Higher values result in stronger rules but fewer in number."
        )
    
    # Run pattern mining
    if st.button("Run Pattern Mining", key="run_pattern_mining"):
        with st.spinner(f"Running {algorithm} algorithm with min_support={min_support} and min_confidence={min_confidence}..."):
            # Store the min_support value in session state
            st.session_state.pattern_min_support = min_support
            # Use the existing algorithm_code variable
            st.session_state.pattern_miner.run(
                algorithm=algorithm_code,
                min_support=min_support,
                min_confidence=min_confidence
            )
            
            # Generate visualizations automatically
            with st.spinner("Generating visualizations..."):
                st.session_state.pattern_miner.visualize_results()
                
            st.success(f"Pattern mining completed! Found {len(st.session_state.pattern_miner.frequent_itemsets)} frequent itemsets and {len(st.session_state.pattern_miner.association_rules) if hasattr(st.session_state.pattern_miner, 'association_rules') else 0} association rules.")
    
    # Display results if available
    if hasattr(st.session_state, 'pattern_miner') and st.session_state.pattern_miner is not None:
        if hasattr(st.session_state.pattern_miner, 'frequent_itemsets') and st.session_state.pattern_miner.frequent_itemsets is not None:
            st.markdown("### Results")
            
            # Metrics overview
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Frequent Itemsets", len(st.session_state.pattern_miner.frequent_itemsets))
            with metric_col2:
                if hasattr(st.session_state.pattern_miner, 'association_rules'):
                    st.metric("Association Rules", len(st.session_state.pattern_miner.association_rules))
                else:
                    st.metric("Association Rules", 0)
            with metric_col3:
                if hasattr(st.session_state, 'pattern_min_support'):
                    st.metric("Min Support Used", f"{st.session_state.pattern_min_support:.4f}")
                else:
                    st.metric("Min Support Used", "0.01")
            
            # Tab view for different result displays
            tab1, tab2, tab3, tab4 = st.tabs(["Frequent Itemsets", "Association Rules", "Visualizations", "Export Results"])
            
            with tab1:
                st.subheader("Top Frequent Itemsets by Support")
                
                # Convert frozenset index to readable strings
                display_itemsets = st.session_state.pattern_miner.frequent_itemsets.copy()
                # Check if the index is already a numeric type, if so, convert to string directly
                if isinstance(display_itemsets.index, pd.RangeIndex) or display_itemsets.index.dtype == 'int64':
                    display_itemsets['itemset_str'] = display_itemsets.index.astype(str)
                else:
                    # Handle frozenset indices properly
                    display_itemsets['itemset_str'] = display_itemsets.index.map(lambda x: ', '.join(sorted(list(x))) if hasattr(x, '__iter__') and not isinstance(x, str) else str(x))
                display_itemsets.sort_values('support', ascending=False, inplace=True)
                
                # Paginated display
                items_per_page = 10
                total_pages = (len(display_itemsets) + items_per_page - 1) // items_per_page
                
                page_number = st.number_input(
                    "Page", min_value=1, max_value=max(1, total_pages), value=1, step=1
                )
                
                start_idx = (page_number - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(display_itemsets))
                
                # Display itemsets
                st.dataframe(
                    display_itemsets.iloc[start_idx:end_idx][['itemset_str', 'support']]
                    .rename(columns={'itemset_str': 'Items', 'support': 'Support'})
                )
                
                st.caption(f"Showing {start_idx+1}-{end_idx} of {len(display_itemsets)} itemsets")
                
                # Filter by specific item
                st.subheader("Filter Itemsets by Item")
                item_to_filter = st.text_input("Enter item name to filter", key="item_filter")
                
                if item_to_filter:
                    filtered_itemsets = display_itemsets[
                        display_itemsets['itemset_str'].str.contains(item_to_filter, case=False)
                    ]
                    
                    if len(filtered_itemsets) > 0:
                        st.write(f"Found {len(filtered_itemsets)} itemsets containing '{item_to_filter}':")
                        st.dataframe(
                            filtered_itemsets[['itemset_str', 'support']]
                            .rename(columns={'itemset_str': 'Items', 'support': 'Support'})
                        )
                    else:
                        st.warning(f"No itemsets found containing '{item_to_filter}'")
            
            with tab2:
                # First, ensure we have the association_rules attribute and it's a DataFrame
                has_rules = (hasattr(st.session_state.pattern_miner, 'association_rules') and 
                            isinstance(st.session_state.pattern_miner.association_rules, pd.DataFrame))
                
                # Get the correct count of rules
                rule_count = len(st.session_state.pattern_miner.association_rules) if has_rules else 0
                
                if rule_count > 0:
                    st.subheader("Top Association Rules by Confidence")
                    
                    # Convert frozenset to strings for display
                    display_rules = st.session_state.pattern_miner.association_rules.copy()
                    display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
                    display_rules['consequents_str'] = display_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
                    display_rules['rule'] = display_rules['antecedents_str'] + ' → ' + display_rules['consequents_str']
                    
                    # Sort by metrics
                    sort_by = st.selectbox(
                        "Sort by metric",
                        ["confidence", "lift", "support"],
                        index=0
                    )
                    
                    display_rules.sort_values(sort_by, ascending=False, inplace=True)
                    
                    # Paginated display
                    rules_per_page = 10
                    total_pages = (len(display_rules) + rules_per_page - 1) // rules_per_page
                    
                    page_number = st.number_input(
                        "Page", min_value=1, max_value=max(1, total_pages), value=1, step=1, key="rules_page"
                    )
                    
                    start_idx = (page_number - 1) * rules_per_page
                    end_idx = min(start_idx + rules_per_page, len(display_rules))
                    
                    # Display rules
                    st.dataframe(
                        display_rules.iloc[start_idx:end_idx][['rule', 'confidence', 'lift', 'support']]
                        .rename(columns={'rule': 'Rule', 'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'})
                    )
                    
                    st.caption(f"Showing {start_idx+1}-{end_idx} of {len(display_rules)} rules")
                    
                    # Filter by specific item in antecedent/consequent
                    st.subheader("Filter Rules by Item")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ant_item = st.text_input("Item in antecedent (left side)", key="ant_filter")
                    
                    with col2:
                        cons_item = st.text_input("Item in consequent (right side)", key="cons_filter")
                    
                    if ant_item or cons_item:
                        filtered_rules = display_rules
                        
                        if ant_item:
                            filtered_rules = filtered_rules[
                                filtered_rules['antecedents_str'].str.contains(ant_item, case=False)
                            ]
                        
                        if cons_item:
                            filtered_rules = filtered_rules[
                                filtered_rules['consequents_str'].str.contains(cons_item, case=False)
                            ]
                        
                        if len(filtered_rules) > 0:
                            filter_desc = []
                            if ant_item:
                                filter_desc.append(f"'{ant_item}' in antecedent")
                            if cons_item:
                                filter_desc.append(f"'{cons_item}' in consequent")
                            
                            st.write(f"Found {len(filtered_rules)} rules with {' and '.join(filter_desc)}:")
                            st.dataframe(
                                filtered_rules[['rule', 'confidence', 'lift', 'support']]
                                .rename(columns={'rule': 'Rule', 'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'})
                            )
                        else:
                            st.warning("No rules found with the specified filters")
                else:
                    st.info("No association rules found with the current parameters. Try lowering the minimum confidence or support thresholds.")
            
            with tab3:
                st.subheader("Visualizations")
                
                # Display visualizations if they exist
                vis_files = [f for f in os.listdir('visualizations') if f.startswith('pattern_')]
                
                if len(vis_files) > 0:
                    # Initialize the selected visualization in session state if not already present
                    if 'pattern_viz_selection' not in st.session_state:
                        st.session_state.pattern_viz_selection = vis_files[0]
                        
                    selected_viz = st.selectbox(
                        "Select visualization",
                        vis_files,
                        index=vis_files.index(st.session_state.pattern_viz_selection),
                        key="pattern_viz_selectbox",
                        on_change=lambda: setattr(st.session_state, 'pattern_viz_selection', st.session_state.pattern_viz_selectbox)
                    )
                    
                    st.image(os.path.join('visualizations', selected_viz), use_container_width=True)
                    
                    # Option to regenerate visualizations
                    if st.button("Regenerate Visualizations"):
                        with st.spinner("Regenerating visualizations..."):
                            st.session_state.pattern_miner.visualize_results()
                        st.success("Visualizations updated!")
                else:
                    st.warning("No visualizations found. Run pattern mining first or check the 'visualizations' directory.")
            
            with tab4:
                st.subheader("Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export Frequent Itemsets to CSV"):
                        with st.spinner("Exporting frequent itemsets..."):
                            # Convert frozensets to strings for CSV export
                            export_itemsets = st.session_state.pattern_miner.frequent_itemsets.copy()
                            export_itemsets['itemsets_str'] = export_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                            export_path = os.path.join('results', 'frequent_itemsets.csv')
                            export_itemsets.to_csv(export_path, index=False)
                            st.success(f"Exported to {export_path}")
                
                with col2:
                    if hasattr(st.session_state.pattern_miner, 'association_rules') and len(st.session_state.pattern_miner.association_rules) > 0:
                        if st.button("Export Association Rules to CSV"):
                            with st.spinner("Exporting association rules..."):
                                # Convert frozensets to strings for CSV export
                                export_rules = st.session_state.pattern_miner.association_rules.copy()
                                export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(x))
                                export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(x))
                                export_path = os.path.join('results', 'association_rules.csv')
                                export_rules.to_csv(export_path, index=False)
                                st.success(f"Exported to {export_path}")
        else:
            st.info("Run pattern mining first to see results and visualizations.")

# Task B: Collaborative Filtering
elif app_mode == "Task B: Collaborative Filtering":
    st.markdown("<h2 class='sub-header'>Task B: Collaborative Filtering</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Collaborative filtering generates personalized product recommendations based on user similarities.
    The system can use various methods: user-based (similar users like similar items), item-based (users who like an item also like similar items),
    or matrix factorization (SVD) which uses latent factors to make predictions.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize collaborative filter if not already done
    if st.session_state.collaborative_filter is None and st.session_state.train_data is not None:
        with st.spinner("Initializing collaborative filtering model..."):
            st.session_state.collaborative_filter = CollaborativeFilter(st.session_state.train_data)
            
            # Prepare data
            st.session_state.collaborative_filter.prepare_data()
    
    # CF is ready
    if st.session_state.collaborative_filter is not None:
        # Sidebar for configuration
        st.sidebar.markdown("### Recommendation Configuration")
        
        # Recommendation method selection
        rec_method = st.sidebar.radio(
            "Recommendation Method",
            ["User-Based", "Item-Based", "SVD (Matrix Factorization)", "Compare All Methods"],
            index=2,
            help="Choose the algorithm for generating recommendations"
        )
        
        # Map to internal method codes
        method_code = {
            "User-Based": "user-based",
            "Item-Based": "item-based",
            "SVD (Matrix Factorization)": "svd",
            "Compare All Methods": "all"
        }[rec_method]
        
        # Number of recommendations
        n_recommendations = st.sidebar.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="How many product recommendations to generate"
        )
        
        # SVD Parameters (if applicable)
        if method_code in ["svd", "all"]:
            n_components = st.sidebar.slider(
                "SVD Components",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Number of latent factors in SVD. Higher values can capture more patterns but may overfit."
            )
        else:
            n_components = 50
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### User Selection")
            
            # Option to select a user or get a random one
            user_selection = st.radio(
                "How would you like to select a user?",
                ["Choose from list", "Enter user ID", "Random user"],
                index=2
            )
            
            user_id = None
            
            if user_selection == "Choose from list":
                # Get sample of users
                all_users = st.session_state.train_data['User_id'].unique()
                sample_users = sorted(random.sample(list(all_users), min(100, len(all_users))))
                
                user_id = st.selectbox(
                    "Select a user",
                    sample_users,
                    index=0,
                    format_func=lambda x: f"User {x}"
                )
            
            elif user_selection == "Enter user ID":
                user_id_input = st.text_input("Enter user ID")
                
                if user_id_input:
                    try:
                        user_id = float(user_id_input)
                        # Check if user exists
                        all_users = set(st.session_state.train_data['User_id'].unique())
                        if user_id not in all_users:
                            st.error(f"User ID {user_id} not found in the dataset")
                            user_id = None
                    except ValueError:
                        st.error("Please enter a valid user ID (number)")
            
            else:  # Random user
                all_users = st.session_state.train_data['User_id'].unique()
                
                if st.button("Select Random User"):
                    user_id = random.choice(all_users)
                    st.success(f"Selected User {user_id}")
            
            # Display user purchase history if a user is selected
            if user_id is not None:
                user_data = st.session_state.train_data[st.session_state.train_data['User_id'] == user_id]
                
                if len(user_data) > 0:
                    st.markdown(f"### Purchase History for User {user_id}")
                    
                    user_items = user_data['itemDescription'].value_counts()
                    
                    st.write(f"This user has purchased {len(user_items)} unique items across {len(user_data)} transactions")
                    
                    # Create bar chart of purchase history
                    fig, ax = plt.subplots(figsize=(10, 5))
                    user_items.head(10).plot(kind='barh', ax=ax)
                    plt.title(f"Top 10 Purchased Items by User {user_id}")
                    plt.xlabel("Purchase Count")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    with st.expander("View Complete Purchase History"):
                        st.dataframe(
                            user_items.reset_index()
                            .rename(columns={'index': 'Item', 'itemDescription': 'Count'})
                        )
        
        with col2:
            st.markdown("### Generate Recommendations")
            
            # Generate button only active if user is selected
            generate_button = st.button(
                "Generate Recommendations",
                disabled=user_id is None
            )
            
            if generate_button and user_id is not None:
                # Compute similarity matrices if not already done
                if method_code in ["user-based", "all"] and not hasattr(st.session_state.collaborative_filter, 'user_similarity'):
                    with st.spinner("Computing user similarity matrix..."):
                        st.session_state.collaborative_filter.compute_similarity(mode='user')
                
                if method_code in ["item-based", "all"] and not hasattr(st.session_state.collaborative_filter, 'item_similarity'):
                    with st.spinner("Computing item similarity matrix..."):
                        st.session_state.collaborative_filter.compute_similarity(mode='item')
                
                if method_code in ["svd", "all"] and not hasattr(st.session_state.collaborative_filter, 'svd_model'):
                    with st.spinner(f"Training SVD model with {n_components} components..."):
                        st.session_state.collaborative_filter.train_svd_model(n_components=n_components)
                
                # Generate recommendations
                if method_code == "all":
                    with st.spinner("Generating recommendations using all methods..."):
                        # User-based recommendations
                        start_time = time.time()
                        user_recommendations = st.session_state.collaborative_filter.generate_recommendations(
                            user_id=user_id, method='user-based', n_recommendations=n_recommendations
                        )
                        user_time = time.time() - start_time
                        
                        # Item-based recommendations
                        start_time = time.time()
                        item_recommendations = st.session_state.collaborative_filter.generate_recommendations(
                            user_id=user_id, method='item-based', n_recommendations=n_recommendations
                        )
                        item_time = time.time() - start_time
                        
                        # SVD-based recommendations
                        start_time = time.time()
                        svd_recommendations = st.session_state.collaborative_filter.generate_recommendations(
                            user_id=user_id, method='svd', n_recommendations=n_recommendations
                        )
                        svd_time = time.time() - start_time
                        
                        # Store in session state
                        st.session_state.recommendations = {
                            'user': user_recommendations,
                            'item': item_recommendations,
                            'svd': svd_recommendations,
                            'execution_times': {
                                'user': user_time,
                                'item': item_time,
                                'svd': svd_time
                            }
                        }
                        
                        # Evaluate recommendations if test data is available
                        if st.session_state.test_data is not None:
                            with st.spinner("Evaluating recommendation quality..."):
                                try:
                                    # Evaluate each method with limited number of users for speed
                                    n_eval_users = 5  # Small number for quick evaluation
                                    
                                    user_metrics = st.session_state.collaborative_filter.evaluate_recommendations(
                                        st.session_state.test_data, method='user-based', n_users=n_eval_users
                                    )
                                    
                                    item_metrics = st.session_state.collaborative_filter.evaluate_recommendations(
                                        st.session_state.test_data, method='item-based', n_users=n_eval_users
                                    )
                                    
                                    svd_metrics = st.session_state.collaborative_filter.evaluate_recommendations(
                                        st.session_state.test_data, method='svd', n_users=n_eval_users
                                    )
                                    
                                    # Store metrics
                                    st.session_state.metrics = {
                                        'user': user_metrics,
                                        'item': item_metrics,
                                        'svd': svd_metrics
                                    }
                                except Exception as e:
                                    st.error(f"Error evaluating recommendations: {e}")
                                    st.session_state.metrics = None
                else:
                    with st.spinner(f"Generating {rec_method} recommendations..."):
                        start_time = time.time()
                        recommendations = st.session_state.collaborative_filter.generate_recommendations(
                            user_id=user_id, method=method_code, n_recommendations=n_recommendations
                        )
                        execution_time = time.time() - start_time
                        
                        # Store in session state
                        st.session_state.recommendations = {
                            method_code: recommendations,
                            'execution_times': {
                                method_code: execution_time
                            }
                        }
                        
                        # Evaluate recommendations if test data is available
                        if st.session_state.test_data is not None:
                            with st.spinner("Evaluating recommendation quality..."):
                                try:
                                    # Evaluate with limited number of users for speed
                                    n_eval_users = 5  # Small number for quick evaluation
                                    
                                    metrics = st.session_state.collaborative_filter.evaluate_recommendations(
                                        st.session_state.test_data, method=method_code, n_users=n_eval_users
                                    )
                                    
                                    # Store metrics
                                    st.session_state.metrics = {
                                        method_code: metrics
                                    }
                                except Exception as e:
                                    st.error(f"Error evaluating recommendations: {e}")
                                    st.session_state.metrics = None
        
        # Display recommendations if available
        if 'recommendations' in st.session_state:
            st.markdown("---")
            st.markdown("## Recommendation Results")
            
            if method_code == "all":
                # Create tabs for different recommendation methods
                user_tab, item_tab, svd_tab, metrics_tab = st.tabs([
                    "User-Based", "Item-Based", "SVD-Based", "Quality Metrics"
                ])
                
                with user_tab:
                    st.markdown("### User-Based Recommendations")
                    try:
                        st.write(f"Generated in {st.session_state.recommendations['execution_times']['user']:.4f} seconds")
                        
                        # Display recommendations
                        rec_data = pd.DataFrame({
                            'Item': list(st.session_state.recommendations['user'].keys()),
                            'Score': list(st.session_state.recommendations['user'].values())
                        })
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        rec_data.head(10).plot(kind='barh', x='Item', y='Score', ax=ax)
                        plt.title(f"Top 10 User-Based Recommendations for User {user_id}")
                        plt.xlabel("Recommendation Score")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display as table
                        st.dataframe(rec_data)
                    except KeyError:
                        st.info("User-based recommendations are not available. Try regenerating recommendations.")
                
                with item_tab:
                    st.markdown("### Item-Based Recommendations")
                    try:
                        st.write(f"Generated in {st.session_state.recommendations['execution_times']['item']:.4f} seconds")
                        
                        # Display recommendations
                        rec_data = pd.DataFrame({
                            'Item': list(st.session_state.recommendations['item'].keys()),
                            'Score': list(st.session_state.recommendations['item'].values())
                        })
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        rec_data.head(10).plot(kind='barh', x='Item', y='Score', ax=ax)
                        plt.title(f"Top 10 Item-Based Recommendations for User {user_id}")
                        plt.xlabel("Recommendation Score")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display as table
                        st.dataframe(rec_data)
                    except KeyError:
                        st.info("Item-based recommendations are not available. Try regenerating recommendations.")
                
                with svd_tab:
                    st.markdown("### SVD-Based Recommendations")
                    try:
                        st.write(f"Generated in {st.session_state.recommendations['execution_times']['svd']:.4f} seconds")
                        
                        # Display recommendations
                        rec_data = pd.DataFrame({
                            'Item': list(st.session_state.recommendations['svd'].keys()),
                            'Score': list(st.session_state.recommendations['svd'].values())
                        })
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        rec_data.head(10).plot(kind='barh', x='Item', y='Score', ax=ax)
                        plt.title(f"Top 10 SVD-Based Recommendations for User {user_id}")
                        plt.xlabel("Recommendation Score")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display as table
                        st.dataframe(rec_data)
                    except KeyError:
                        st.info("SVD-based recommendations are not available. Try regenerating recommendations.")
                
                with metrics_tab:
                    if 'metrics' in st.session_state and st.session_state.metrics is not None:
                        st.markdown("### Quality Metrics Comparison")
                        
                        # Make sure all expected keys exist in metrics
                        metrics_keys = ['user', 'item', 'svd']
                        if all(key in st.session_state.metrics for key in metrics_keys):
                            # Create metrics comparison dataframe
                            try:
                                metrics_df = pd.DataFrame({
                                    'User-Based': [
                                        st.session_state.metrics['user']['hit_rate'] if 'hit_rate' in st.session_state.metrics['user'] else 0,
                                        st.session_state.metrics['user']['average_precision'] if 'average_precision' in st.session_state.metrics['user'] else 0,
                                        st.session_state.metrics['user']['coverage'] if 'coverage' in st.session_state.metrics['user'] else 0,
                                        st.session_state.metrics['user']['diversity'] if 'diversity' in st.session_state.metrics['user'] else 0
                                    ],
                                    'Item-Based': [
                                        st.session_state.metrics['item']['hit_rate'] if 'hit_rate' in st.session_state.metrics['item'] else 0,
                                        st.session_state.metrics['item']['average_precision'] if 'average_precision' in st.session_state.metrics['item'] else 0,
                                        st.session_state.metrics['item']['coverage'] if 'coverage' in st.session_state.metrics['item'] else 0,
                                        st.session_state.metrics['item']['diversity'] if 'diversity' in st.session_state.metrics['item'] else 0
                                    ],
                                    'SVD-Based': [
                                        st.session_state.metrics['svd']['hit_rate'] if 'hit_rate' in st.session_state.metrics['svd'] else 0,
                                        st.session_state.metrics['svd']['average_precision'] if 'average_precision' in st.session_state.metrics['svd'] else 0,
                                        st.session_state.metrics['svd']['coverage'] if 'coverage' in st.session_state.metrics['svd'] else 0,
                                        st.session_state.metrics['svd']['diversity'] if 'diversity' in st.session_state.metrics['svd'] else 0
                                    ]
                                }, index=['Hit Rate', 'Precision', 'Coverage', 'Diversity'])
                                
                                # Display metrics as a table
                                st.dataframe(metrics_df.style.highlight_max(axis=1))
                                
                                # Create radar chart
                                labels = ['Hit Rate', 'Precision', 'Coverage', 'Diversity']
                                methods = ['User-Based', 'Item-Based', 'SVD-Based']
                                
                                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                                angles += angles[:1]  # Close the polygon
                                
                                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                                
                                for i, method in enumerate(methods):
                                    values = metrics_df[method].tolist()
                                    values += values[:1]  # Close the polygon
                                    ax.plot(angles, values, linewidth=2, label=method)
                                    ax.fill(angles, values, alpha=0.1)
                                
                                ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                                ax.set_ylim(0, max(metrics_df.max()) * 1.1)
                                ax.grid(True)
                                ax.legend(loc='upper right')
                                plt.title('Recommendation Quality Metrics Comparison')
                                
                                st.pyplot(fig)
                                
                                st.markdown("""
                                ### Metrics Explanation
                                - **Hit Rate**: Proportion of recommended items that users actually purchase
                                - **Precision**: Accuracy of recommendations
                                - **Coverage**: Range of items that can be recommended
                                - **Diversity**: Variety in recommendations
                                """)
                            except Exception as e:
                                st.error(f"Error displaying metrics: {e}")
                                st.info("Try regenerating recommendations to fix this issue.")
                        else:
                            st.info("Complete metrics for all recommendation methods are not available. Try regenerating all recommendations.")
                    
                    # Visual comparison of recommendations
                    st.markdown("### Visual Comparison of Recommendations")
                    
                    # Create a merged dataframe of all recommendations
                    all_items = set()
                    for method in ['user', 'item', 'svd']:
                        all_items.update(st.session_state.recommendations[method].keys())
                    
                    comparison_df = pd.DataFrame(index=sorted(all_items))
                    
                    for method, name in zip(['user', 'item', 'svd'], ['User-Based', 'Item-Based', 'SVD-Based']):
                        method_scores = pd.Series(st.session_state.recommendations[method], name=name)
                        comparison_df = comparison_df.join(method_scores)
                    
                    comparison_df.fillna(0, inplace=True)
                    
                    # Create heatmap of recommendations
                    plt.figure(figsize=(12, min(20, max(8, len(all_items) / 5))))
                    sns.heatmap(comparison_df.head(20), annot=True, cmap="YlGnBu", fmt=".2f")
                    plt.title(f"Top Item Recommendations Across Methods for User {user_id}")
                    plt.tight_layout()
                    
                    st.pyplot(plt)
                    
                    # Display intersection as a Venn diagram
                    from matplotlib_venn import venn3
                    
                    # Get top N items from each method
                    top_n = 10
                    user_top = set(list(st.session_state.recommendations['user'].keys())[:top_n])
                    item_top = set(list(st.session_state.recommendations['item'].keys())[:top_n])
                    svd_top = set(list(st.session_state.recommendations['svd'].keys())[:top_n])
                    
                    # Create Venn diagram
                    plt.figure(figsize=(8, 8))
                    venn = venn3([user_top, item_top, svd_top], ['User-Based', 'Item-Based', 'SVD-Based'])
                    plt.title(f"Overlap of Top {top_n} Recommendations Across Methods")
                    
                    st.pyplot(plt)
                    
                    # Show recommendations in common
                    common_all = user_top.intersection(item_top).intersection(svd_top)
                    if common_all:
                        st.markdown(f"**Items recommended by all methods:** {', '.join(common_all)}")
            
            else:
                # Single method recommendations
                st.markdown(f"### {rec_method} Recommendations")
                st.write(f"Generated in {st.session_state.recommendations['execution_times'][method_code]:.4f} seconds")
                
                # Display recommendations
                rec_data = pd.DataFrame({
                    'Item': list(st.session_state.recommendations[method_code].keys()),
                    'Score': list(st.session_state.recommendations[method_code].values())
                })
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                rec_data.head(10).plot(kind='barh', x='Item', y='Score', ax=ax, color='#1E88E5')
                plt.title(f"Top 10 {rec_method} Recommendations for User {user_id}")
                plt.xlabel("Recommendation Score")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display as table
                st.dataframe(rec_data)
                
                # Display metrics if available
                if 'metrics' in st.session_state and st.session_state.metrics is not None:
                    st.markdown("### Quality Metrics")
                    
                    # Create metrics display in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Hit Rate",
                            f"{st.session_state.metrics[method_code]['hit_rate']:.4f}",
                            help="Proportion of recommended items that users actually purchase"
                        )
                    
                    with col2:
                        st.metric(
                            "Precision",
                            f"{st.session_state.metrics[method_code]['average_precision']:.4f}",
                            help="Accuracy of recommendations"
                        )
                    
                    with col3:
                        st.metric(
                            "Coverage",
                            f"{st.session_state.metrics[method_code]['coverage']:.4f}",
                            help="Range of items that can be recommended"
                        )
                    
                    with col4:
                        st.metric(
                            "Diversity",
                            f"{st.session_state.metrics[method_code]['diversity']:.4f}",
                            help="Variety in recommendations"
                        )
            
            # Generate visualizations
            if st.button("Generate Additional Visualizations"):
                with st.spinner("Generating visualizations..."):
                    # Call the visualization function
                    st.session_state.collaborative_filter.visualize_results()
                
                st.success("Visualizations generated successfully! Check the 'visualizations' directory.")
                
                # Display some visualizations if they exist
                vis_files = [f for f in os.listdir('visualizations') if f.startswith(('recommendation_', 'user_'))]
                
                if vis_files:
                    # Initialize the selected visualization in session state if not already present
                    if 'cf_viz_selection' not in st.session_state:
                        st.session_state.cf_viz_selection = vis_files[0]
                        
                    selected_viz = st.selectbox(
                        "Select visualization to display",
                        vis_files,
                        index=vis_files.index(st.session_state.cf_viz_selection),
                        key="cf_viz_selectbox",
                        on_change=lambda: setattr(st.session_state, 'cf_viz_selection', st.session_state.cf_viz_selectbox)
                    )
                    
                    st.image(os.path.join('visualizations', selected_viz), use_container_width=True)
    else:
        st.warning("Collaborative filtering model not initialized. Please check that the dataset is loaded correctly.")

# Task C: Integrated System
elif app_mode == "Task C: Integrated System":
    st.markdown("<h2 class='sub-header'>Task C: Integrated System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    The integrated system combines pattern mining and collaborative filtering to create enhanced recommendations.
    Pattern mining identifies complementary products based on purchase associations, while collaborative filtering
    captures user preferences and similarities. The hybrid approach leverages the strengths of both methods.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize integrated system if not already done
    if st.session_state.integrated_system is None and st.session_state.train_data is not None and st.session_state.test_data is not None:
        with st.spinner("Initializing integrated system..."):
            st.session_state.integrated_system = IntegratedSystem(st.session_state.train_data, st.session_state.test_data)
    
    # Integrated system is ready
    if st.session_state.integrated_system is not None:
        # Sidebar for configuration
        st.sidebar.markdown("### Integration Configuration")
        
        # Pattern mining parameters
        st.sidebar.markdown("#### Pattern Mining Parameters")
        
        algorithm = st.sidebar.radio(
            "Mining Algorithm",
            ["Apriori", "FP-Growth"],
            index=1,
            help="Algorithm for pattern mining"
        )
        algorithm_code = 'apriori' if algorithm == "Apriori" else 'fpgrowth'
        
        min_support = st.sidebar.slider(
            "Minimum Support",
            min_value=0.001,
            max_value=0.1,
            value=0.005,
            step=0.001,
            format="%.3f",
            help="Minimum support threshold for pattern mining"
        )
        
        # Collaborative filtering parameters
        st.sidebar.markdown("#### Collaborative Filtering Parameters")
        
        n_components = st.sidebar.slider(
            "SVD Components",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of latent factors in SVD"
        )
        
        # Integration parameters
        st.sidebar.markdown("#### Integration Parameters")
        
        with_patterns = st.sidebar.checkbox(
            "Include Pattern Mining",
            value=True,
            help="Include pattern mining in the hybrid recommendations"
        )
        
        n_recommendations = st.sidebar.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="How many product recommendations to generate"
        )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # System setup
            st.markdown("### System Setup")
            
            setup_tab1, setup_tab2 = st.tabs(["Pattern Mining", "Collaborative Filtering"])
            
            with setup_tab1:
                # Pattern mining setup
                if not hasattr(st.session_state.integrated_system.pattern_miner, 'frequent_itemsets') or st.session_state.integrated_system.pattern_miner.frequent_itemsets is None:
                    st.info("Pattern mining not initialized. Click 'Run Pattern Mining' to prepare this component.")
                    
                    pattern_mining_button = st.button("Run Pattern Mining")
                    
                    if pattern_mining_button:
                        with st.spinner(f"Running {algorithm.upper()} algorithm with min_support={min_support}..."):
                            start_time = time.time()
                            # Store min_support in session state
                            st.session_state.pattern_min_support = min_support
                            # Use the existing algorithm_code variable
                            st.session_state.integrated_system.mine_patterns(
                                min_support=min_support,
                                algorithm=algorithm_code
                            )
                            pattern_mining_time = time.time() - start_time
                            
                            if hasattr(st.session_state.integrated_system.pattern_miner, 'frequent_itemsets'):
                                n_itemsets = len(st.session_state.integrated_system.pattern_miner.frequent_itemsets)
                                n_rules = len(st.session_state.integrated_system.pattern_miner.association_rules) if hasattr(st.session_state.integrated_system.pattern_miner, 'association_rules') else 0
                                
                                st.success(f"Pattern mining completed in {pattern_mining_time:.2f} seconds. Found {n_itemsets} frequent itemsets and {n_rules} association rules.")
                else:
                    n_itemsets = len(st.session_state.integrated_system.pattern_miner.frequent_itemsets)
                    n_rules = len(st.session_state.integrated_system.pattern_miner.association_rules) if hasattr(st.session_state.integrated_system.pattern_miner, 'association_rules') else 0
                    
                    st.success(f"Pattern mining ready. Found {n_itemsets} frequent itemsets and {n_rules} association rules.")
                    
                    # Option to re-run with different parameters
                    rerun_patterns = st.button("Re-run Pattern Mining")
                    
                    if rerun_patterns:
                        with st.spinner(f"Re-running {algorithm.upper()} algorithm with min_support={min_support}..."):
                            start_time = time.time()
                            # Store min_support in session state
                            st.session_state.pattern_min_support = min_support
                            # Use the existing algorithm_code variable
                            st.session_state.integrated_system.mine_patterns(
                                min_support=min_support,
                                algorithm=algorithm_code
                            )
                            pattern_mining_time = time.time() - start_time
                            
                            n_itemsets = len(st.session_state.integrated_system.pattern_miner.frequent_itemsets)
                            n_rules = len(st.session_state.integrated_system.pattern_miner.association_rules) if hasattr(st.session_state.integrated_system.pattern_miner, 'association_rules') else 0
                            
                            st.success(f"Pattern mining updated in {pattern_mining_time:.2f} seconds. Found {n_itemsets} frequent itemsets and {n_rules} association rules.")
            
            with setup_tab2:
                # Collaborative filtering setup
                if not hasattr(st.session_state.integrated_system.collaborative_filter, 'user_item_matrix') or st.session_state.integrated_system.collaborative_filter.user_item_matrix is None:
                    st.info("Collaborative filtering not initialized. Click 'Prepare Collaborative Filtering' to set up this component.")
                    
                    cf_setup_button = st.button("Prepare Collaborative Filtering")
                    
                    if cf_setup_button:
                        with st.spinner("Preparing collaborative filtering..."):
                            start_time = time.time()
                            st.session_state.integrated_system.prepare_collaborative_filtering()
                            cf_time = time.time() - start_time
                            
                            if hasattr(st.session_state.integrated_system.collaborative_filter, 'user_item_matrix'):
                                n_users = st.session_state.integrated_system.collaborative_filter.user_item_matrix.shape[0]
                                n_items = st.session_state.integrated_system.collaborative_filter.user_item_matrix.shape[1]
                                
                                st.success(f"Collaborative filtering prepared in {cf_time:.2f} seconds. Matrix size: {n_users} users × {n_items} items.")
                else:
                    n_users = st.session_state.integrated_system.collaborative_filter.user_item_matrix.shape[0]
                    n_items = st.session_state.integrated_system.collaborative_filter.user_item_matrix.shape[1]
                    
                    st.success(f"Collaborative filtering ready. Matrix size: {n_users} users × {n_items} items.")
                    
                    # Option to retrain SVD model
                    retrain_svd = st.button("Retrain SVD Model")
                    
                    if retrain_svd:
                        with st.spinner(f"Training SVD model with {n_components} components..."):
                            start_time = time.time()
                            st.session_state.integrated_system.collaborative_filter.train_svd_model(n_components=n_components)
                            svd_time = time.time() - start_time
                            
                            st.success(f"SVD model trained in {svd_time:.2f} seconds with {n_components} components.")
            
            # User selection
            st.markdown("### User Selection")
            
            # Option to select a user or get a random one
            user_selection = st.radio(
                "How would you like to select a user?",
                ["Choose from list", "Enter user ID", "Random user"],
                index=2
            )
            
            user_id = None
            
            if user_selection == "Choose from list":
                # Get sample of users
                all_users = st.session_state.train_data['User_id'].unique()
                sample_users = sorted(random.sample(list(all_users), min(100, len(all_users))))
                
                user_id = st.selectbox(
                    "Select a user",
                    sample_users,
                    index=0,
                    format_func=lambda x: f"User {x}",
                    key="user_analysis_selectbox"
                )
            
            elif user_selection == "Enter user ID":
                user_id_input = st.text_input("Enter user ID", key="user_analysis_input")
                
                if user_id_input:
                    try:
                        user_id = float(user_id_input)
                        # Check if user exists
                        all_users = set(st.session_state.train_data['User_id'].unique())
                        if user_id not in all_users:
                            st.error(f"User ID {user_id} not found in the dataset")
                            user_id = None
                    except ValueError:
                        st.error("Please enter a valid user ID (number)")
            
            else:  # Random user
                all_users = list(st.session_state.train_data['User_id'].unique())
                
                if st.button("Select Random User", key="user_analysis_random"):
                    user_id = random.choice(all_users)
                    st.success(f"Selected User {user_id}")
            
            # Display user purchase history and analysis if a user is selected
            if user_id is not None:
                st.markdown("---")
                st.markdown(f"#### Analysis for User {user_id}")
                
                user_data = st.session_state.train_data[st.session_state.train_data['User_id'] == user_id]
                
                if len(user_data) > 0:
                    # Basic metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Purchases", len(user_data))
                    
                    with col2:
                        st.metric("Unique Products", user_data['itemDescription'].nunique())
                    
                    with col3:
                        # Calculate shopping frequency if date column exists
                        if 'Date' in user_data.columns:
                            date_range = pd.to_datetime(user_data['Date'], dayfirst=True).max() - pd.to_datetime(user_data['Date'], dayfirst=True).min()
                            days = max(1, date_range.days)
                            frequency = len(user_data) / days
                            st.metric("Shopping Frequency", f"{frequency:.2f} items/day")
                        else:
                            st.metric("Shopping Frequency", "N/A")
                    
                    # Tabs for different visualizations
                    user_viz_tabs = st.tabs(["Purchase History", "Product Categories", "Shopping Patterns"])
                    
                    with user_viz_tabs[0]:
                        st.markdown("##### Purchase History")
                        
                        # Product purchase counts
                        item_counts = user_data['itemDescription'].value_counts()
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, min(12, max(5, len(item_counts[:15]) / 2))))
                        item_counts[:15].plot(kind='barh', ax=ax)
                        plt.title(f"Top 15 Products Purchased by User {user_id}")
                        plt.xlabel("Purchase Count")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show full purchase history
                        with st.expander("View Complete Purchase History"):
                            st.dataframe(
                                item_counts.reset_index()
                                .rename(columns={'index': 'Product', 'itemDescription': 'Count'})
                            )
                    
                    with user_viz_tabs[1]:
                        st.markdown("##### Product Categories")
                        
                        # Check if category data is available
                        if 'category' in user_data.columns:
                            # Aggregate by category
                            category_counts = user_data['category'].value_counts()
                            
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(8, 8))
                            category_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                            plt.title(f"Product Categories Purchased by User {user_id}")
                            plt.ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            # Extract product categories if not in dataset
                            # This is a simplification - in a real app you might use NLP or a product database
                            # to properly categorize products
                            
                            # Try to extract categories from product names
                            def extract_category(product_name):
                                if pd.isna(product_name):
                                    return "Unknown"
                                
                                # Simple category extraction based on keywords
                                product_name = safe_str(product_name).lower()
                                
                                categories = {
                                    "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
                                    "meat": ["meat", "beef", "chicken", "pork", "steak", "sausage"],
                                    "produce": ["fruit", "vegetable", "apple", "banana", "tomato", "lettuce", "fresh"],
                                    "bakery": ["bread", "cake", "pastry", "bun", "roll"],
                                    "beverages": ["water", "soda", "juice", "coffee", "tea", "drink"],
                                    "snacks": ["chip", "crisp", "snack", "chocolate", "candy", "sweet"],
                                    "canned": ["can", "canned", "soup", "preserved"],
                                    "frozen": ["frozen", "ice cream", "freezer"],
                                    "household": ["paper", "cleaner", "detergent", "soap", "household"]
                                }
                                
                                for category, keywords in categories.items():
                                    for keyword in keywords:
                                        if keyword in product_name:
                                            return category
                                
                                return "Other"
                            
                            # Add categories to user_data
                            product_categories = user_data['itemDescription'].apply(extract_category)
                            category_counts = product_categories.value_counts()
                            
                            # Create pie chart
                            fig, ax = plt.subplots(figsize=(8, 8))
                            category_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                            plt.title(f"Estimated Product Categories for User {user_id}")
                            plt.ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.caption("Note: Categories are estimated based on product names and may not be 100% accurate.")
                    
                    with user_viz_tabs[2]:
                        st.markdown("##### Shopping Patterns")
                        
                        # Analyze frequently bought together products for this user
                        if len(user_data) > 5:  # Need enough data for meaningful patterns
                            st.markdown("**Frequently Bought Together Products**")
                            
                            # Get itemsets from transactions
                            from mlxtend.preprocessing import TransactionEncoder
                            from mlxtend.frequent_patterns import apriori, association_rules
                            
                            # Function to extract user transactions
                            def extract_user_transactions(user_data):
                                # Group by transaction if a transaction ID exists
                                if 'InvoiceNo' in user_data.columns:
                                    return user_data.groupby('InvoiceNo')['itemDescription'].apply(list).tolist()
                                else:
                                    # If no transaction ID, you can use date for grouping
                                    if 'Date' in user_data.columns:
                                        return user_data.groupby('Date')['itemDescription'].apply(list).tolist()
                                    else:
                                        # Without transaction grouping, just return individual items as single-item transactions
                                        return [[item] for item in user_data['itemDescription'].tolist()]
                            
                            # Get user transactions
                            user_transactions = extract_user_transactions(user_data)
                            
                            # Only proceed if we have multi-item transactions
                            multi_item_transactions = [t for t in user_transactions if len(t) > 1]
                            
                            if len(multi_item_transactions) > 2:
                                # Convert to binary representation
                                te = TransactionEncoder()
                                te_ary = te.fit(multi_item_transactions).transform(multi_item_transactions)
                                df = pd.DataFrame(te_ary, columns=te.columns_)
                                
                                # Find frequent itemsets
                                try:
                                    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
                                    
                                    if len(frequent_itemsets) > 0:
                                        # Convert itemsets to readable format
                                        frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(x))
                                        
                                        # Display the top itemsets
                                        st.dataframe(
                                            frequent_itemsets.sort_values('support', ascending=False)
                                            .head(10)[['itemsets_str', 'support']]
                                            .rename(columns={'itemsets_str': 'Products Bought Together', 'support': 'Support'})
                                        )
                                        
                                        # Try to generate rules
                                        if len(frequent_itemsets) > 1:
                                            try:
                                                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
                                                
                                                if len(rules) > 0:
                                                    # Convert to readable format
                                                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(x))
                                                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(x))
                                                    rules['rule'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
                                                    
                                                    st.markdown("**Product Association Rules:**")
                                                    st.dataframe(
                                                        rules.sort_values('confidence', ascending=False)
                                                        .head(10)[['rule', 'confidence', 'lift', 'support']]
                                                        .rename(columns={'rule': 'Rule', 'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'})
                                                    )
                                            except Exception as e:
                                                st.info("Could not generate association rules from the user's purchase patterns.")
                                    else:
                                        st.info("No frequent itemsets found in the user's purchase patterns.")
                                except Exception as e:
                                    st.info(f"Could not analyze frequently bought together products: {e}")
                            else:
                                st.info("Not enough multi-item purchase records to find patterns.")
                        else:
                            st.info("Not enough purchase data to analyze patterns.")
                        
                        # Purchase timeline if date data is available
                        if 'Date' in user_data.columns:
                            st.markdown("**Purchase Timeline**")
                            
                            # Convert to datetime if not already
                            if not pd.api.types.is_datetime64_any_dtype(user_data['Date']):
                                user_data['Date'] = pd.to_datetime(user_data['Date'], dayfirst=True)
                            
                            # Group by date and count purchases
                            timeline = user_data.groupby(user_data['Date'].dt.date).size().reset_index(name='purchases')
                            
                            # Create timeline chart
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(timeline['Date'], timeline['purchases'], marker='o')
                            plt.title(f"Purchase Timeline for User {user_id}")
                            plt.xlabel("Date")
                            plt.ylabel("Number of Items Purchased")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.warning(f"No purchase data found for User {user_id}")
            
            st.markdown("---")
            st.markdown("#### User Cohort Analysis")
            
            # Allow segmenting users by purchase behavior
            st.markdown("Group users by purchase behavior to identify different customer segments:")
            
            if st.button("Run Cohort Analysis"):
                with st.spinner("Analyzing user cohorts..."):
                    # Calculate purchase metrics for each user
                    user_metrics = st.session_state.train_data.groupby('User_id').agg(
                        total_purchases=('itemDescription', 'count'),
                        unique_products=('itemDescription', 'nunique')
                    ).reset_index()
                    
                    # Simple K-means clustering for segmentation
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans
                    
                    # Prepare data for clustering
                    X = user_metrics[['total_purchases', 'unique_products']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Determine optimal number of clusters using elbow method
                    inertias = []
                    k_range = range(2, min(10, len(user_metrics)))
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(X_scaled)
                        inertias.append(kmeans.inertia_)
                    
                    # Plot elbow method results
                    fig, ax = plt.subplots(figsize=(8, 5))
                    plt.plot(k_range, inertias, 'bo-')
                    plt.xlabel('Number of clusters')
                    plt.ylabel('Inertia')
                    plt.title('Elbow Method for Optimal k')
                    st.pyplot(fig)
                    
                    # Choose k based on elbow method (or select a reasonable default)
                    k = 4  # You can make this dynamic based on the elbow plot
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    user_metrics['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Create descriptive labels for clusters
                    cluster_descriptions = {
                        0: "Low purchase frequency, low variety",
                        1: "Low purchase frequency, high variety",
                        2: "High purchase frequency, low variety",
                        3: "High purchase frequency, high variety"
                    }
                    
                    # Update descriptions based on actual cluster centers
                    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                                 columns=['total_purchases', 'unique_products'])
                    
                    for i in range(len(cluster_centers)):
                        purchases = "High" if cluster_centers.iloc[i]['total_purchases'] > cluster_centers['total_purchases'].median() else "Low"
                        variety = "High" if cluster_centers.iloc[i]['unique_products'] > cluster_centers['unique_products'].median() else "Low"
                        cluster_descriptions[i] = f"{purchases} purchase frequency, {variety} variety"
                    
                    # Add descriptions to the dataframe
                    user_metrics['segment'] = user_metrics['cluster'].map(lambda x: f"Segment {x+1}: {cluster_descriptions[x]}")
                    
                    # Display results
                    st.markdown("#### Customer Segments")
                    
                    # Show segment statistics
                    segment_stats = user_metrics.groupby('segment').agg(
                        user_count=('User_id', 'count'),
                        avg_purchases=('total_purchases', 'mean'),
                        avg_unique_products=('unique_products', 'mean')
                    ).reset_index()
                    
                    st.dataframe(segment_stats)
                    
                    # Visualize the clusters
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(user_metrics['total_purchases'], user_metrics['unique_products'], 
                                       c=user_metrics['cluster'], cmap='viridis', alpha=0.6)
                    
                    # Add cluster centers
                    ax.scatter(cluster_centers['total_purchases'], cluster_centers['unique_products'], 
                              s=200, marker='X', c='red', label='Cluster centers')
                    
                    # Add legend with segment descriptions
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                label=f"Segment {i+1}: {desc}",
                                                markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                                     for i, desc in cluster_descriptions.items()]
                    
                    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
                    
                    plt.title('Customer Segmentation by Purchase Behavior')
                    plt.xlabel('Total Purchases')
                    plt.ylabel('Unique Products')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show which segment the selected user belongs to (if any)
                    if user_id is not None:
                        user_segment = user_metrics[user_metrics['User_id'] == user_id]['segment'].values
                        if len(user_segment) > 0:
                            st.success(f"User {user_id} belongs to {user_segment[0]}")
                            
                            # Show similar users in the same segment
                            same_segment_users = user_metrics[user_metrics['segment'] == user_segment[0]]['User_id'].tolist()
                            same_segment_users.remove(user_id)  # Remove the current user
                            
                            if same_segment_users:
                                st.markdown(f"**Similar users in the same segment:**")
                                st.write(f"Sample of 5 similar users: {', '.join(map(str, same_segment_users[:5]))}")
                        else:
                            st.info(f"User {user_id} was not found in the segmentation.")
            
            st.markdown("---")
            st.markdown("#### Download User Analysis Report")
            
            # Option to download a PDF/CSV report of the analysis
            if user_id is not None:
                if st.button("Generate User Analysis CSV"):
                    if len(user_data) > 0:
                        # Create report dataframes
                        user_summary = pd.DataFrame({
                            'User_ID': [user_id],
                            'Total_Purchases': [len(user_data)],
                            'Unique_Products': [user_data['itemDescription'].nunique()],
                        })
                        
                        # Convert to CSV
                        user_summary_csv = user_summary.to_csv(index=False)
                        purchases_csv = user_data.to_csv(index=False)
                        
                        # Combine reports
                        csv_report = "USER SUMMARY\n" + user_summary_csv + "\n\nPURCHASE HISTORY\n" + purchases_csv
                        
                        # Create download button
                        st.download_button(
                            label="Download User Analysis CSV",
                            data=csv_report,
                            file_name=f"user_{user_id}_analysis.csv",
                            mime="text/csv",
                        )

# Task D: Interactive Analysis
elif app_mode == "Task D: Interactive Analysis":
    st.markdown("<h2 class='sub-header'>Task D: Interactive Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This interactive analysis tool allows you to explore the grocery store dataset in depth
    and gain insights into customer behavior, product relationships, and recommendation performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different analysis types
    analysis_tabs = st.tabs([
        "User Analysis", 
        "Product Analysis", 
        "Recommendation Comparison",
        "Custom Queries"
    ])
    
    with analysis_tabs[0]:
        st.markdown("### User Analysis")
        
        # User selection
        st.markdown("#### Select User for Analysis")
        
        # Option to select a user or get a random one
        user_selection = st.radio(
            "How would you like to select a user?",
            ["Choose from list", "Enter user ID", "Random user"],
            index=2,
            key="user_analysis_selection"
        )
        
        selected_user = None
        
        # Logic for user selection
        if user_selection == "Choose from list":
            # Get a manageable list of users (top users by purchase count)
            top_users = st.session_state.train_data['User_id'].value_counts().head(100).index.tolist()
            selected_user = st.selectbox("Select a user", options=top_users, key="user_analysis_dropdown")
            
        elif user_selection == "Enter user ID":
            user_id_input = st.text_input("Enter user ID", key="user_analysis_input")
            if user_id_input:
                try:
                    user_id_input = float(user_id_input)
                    if user_id_input in st.session_state.train_data['User_id'].values:
                        selected_user = user_id_input
                    else:
                        st.warning(f"User ID {user_id_input} not found in the dataset")
                except ValueError:
                    st.warning("Please enter a valid user ID")
                    
        elif user_selection == "Random user":
            if st.button("Get Random User", key="user_analysis_random"):
                selected_user = random.choice(st.session_state.train_data['User_id'].unique())
                st.success(f"Selected User {selected_user}")
        
        # Display user purchase history and analysis if a user is selected
        if selected_user is not None:
            # User purchase history
            user_data = st.session_state.train_data[st.session_state.train_data['User_id'] == selected_user]
            
            if len(user_data) > 0:
                st.markdown("---")
                st.markdown(f"#### Purchase History for User {selected_user}")
                
                # Basic user metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Purchases", len(user_data))
                
                with col2:
                    st.metric("Unique Products", user_data['itemDescription'].nunique())
                
                with col3:
                    # Average purchases per day/week if date information is available
                    if 'Date' in user_data.columns:
                        date_range = pd.to_datetime(user_data['Date'], dayfirst=True).max() - pd.to_datetime(user_data['Date'], dayfirst=True).min()
                        days = max(1, date_range.days)
                        frequency = len(user_data) / days
                        st.metric("Shopping Frequency", f"{frequency:.2f} items/day")
                    else:
                        st.metric("Shopping Pattern", "N/A")
                
                # Top purchased items
                st.markdown("#### Top Purchased Items")
                top_items = user_data['itemDescription'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_items.values, y=top_items.index, ax=ax)
                ax.set_title(f"Top 10 Items Purchased by User {selected_user}")
                ax.set_xlabel("Number of Purchases")
                st.pyplot(fig)
                
                # Add download button for user purchase history
                user_report = user_data[['User_id', 'itemDescription', 'Date']].copy() if 'Date' in user_data.columns else user_data[['User_id', 'itemDescription']].copy()
                
                st.markdown("#### Download User Analysis")
                create_download_button(
                    user_report,
                    f"user_{selected_user}_purchase_history.csv",
                    "Download Purchase History"
                )
                
                # Find similar users if collaborative filter is initialized
                st.markdown("---")
                st.markdown("#### Similar Users")
                
                # Initialize collaborative filter if not already done
                if st.session_state.collaborative_filter is None:
                    with st.spinner("Initializing collaborative filter..."):
                        st.session_state.collaborative_filter = CollaborativeFilter(st.session_state.train_data)
                        st.session_state.collaborative_filter.prepare_data()
                
                # Check if user similarity matrix exists, if not compute it
                if not hasattr(st.session_state.collaborative_filter, 'user_similarity') or st.session_state.collaborative_filter.user_similarity is None:
                    with st.spinner("Computing user similarity matrix..."):
                        st.session_state.collaborative_filter.compute_similarity(mode='user')
                
                # Make sure user similarity matrix is computed and the selected user exists in it
                if hasattr(st.session_state.collaborative_filter, 'user_similarity') and st.session_state.collaborative_filter.user_similarity is not None:
                    if selected_user in st.session_state.collaborative_filter.user_similarity.index:
                        similar_users = st.session_state.collaborative_filter.user_similarity.loc[selected_user].sort_values(ascending=False)
                        
                        # Exclude the user itself
                        if selected_user in similar_users.index:
                            similar_users = similar_users.drop(selected_user)
                        
                        similar_df = similar_users.head(10).reset_index()
                        similar_df.columns = ['User_id', 'Similarity']
                        
                        st.write("Top 10 users with similar purchase patterns:")
                        st.dataframe(similar_df)
                        
                        # Visualization of similar users
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Similarity', y='User_id', data=similar_df.head(10), ax=ax)
                        ax.set_title(f"Top 10 Users Similar to User {selected_user}")
                        ax.set_xlabel("Similarity Score")
                        ax.set_ylabel("User ID")
                        st.pyplot(fig)
                    else:
                        st.info(f"User {selected_user} not found in the similarity matrix. This may happen if the user has no common purchases with others.")
                else:
                    st.warning("User similarity matrix is not available. Please try again or check the collaborative filtering system.")
            else:
                st.warning(f"No purchase history found for User {selected_user}")

    with analysis_tabs[1]:
        st.markdown("### Product Analysis")
        
        # Product selection
        st.markdown("#### Select Product for Analysis")
        
        # Option to select a product or get a random one
        product_selection = st.radio(
            "How would you like to select a product?",
            ["Choose from list", "Enter product name", "Random product"],
            index=0,
            key="product_analysis_selection"
        )
        
        selected_product = None
        
        # Logic for product selection
        if product_selection == "Choose from list":
            # Get most common products
            top_products = st.session_state.train_data['itemDescription'].value_counts().head(100).index.tolist()
            selected_product = st.selectbox("Select a product", options=top_products, key="product_analysis_dropdown")
            
        elif product_selection == "Enter product name":
            # Get partial matches to typed product
            product_input = st.text_input("Enter product name", key="product_analysis_input")
            
            if product_input:
                matching_products = st.session_state.train_data['itemDescription'].str.contains(product_input, case=False)
                if any(matching_products):
                    matching_options = st.session_state.train_data[matching_products]['itemDescription'].unique()
                    selected_product = st.selectbox("Select from matching products:", options=matching_options, key="product_analysis_matches")
                else:
                    st.warning(f"No products found matching '{product_input}'")
                    
        elif product_selection == "Random product":
            if st.button("Get Random Product", key="product_analysis_random"):
                selected_product = random.choice(st.session_state.train_data['itemDescription'].unique())
                st.success(f"Selected Product: {selected_product}")
        
        # Display product analysis if a product is selected
        if selected_product is not None:
            product_data = st.session_state.train_data[st.session_state.train_data['itemDescription'] == selected_product]
            
            if len(product_data) > 0:
                st.markdown("---")
                st.markdown(f"#### Analysis for '{selected_product}'")
                
                # Product metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Purchases", len(product_data))
                
                with col2:
                    st.metric("Unique Buyers", product_data['User_id'].nunique())
                
                with col3:
                    avg_per_buyer = len(product_data) / product_data['User_id'].nunique()
                    st.metric("Avg. Purchases per Buyer", f"{avg_per_buyer:.2f}")
                
                # Top buyers of this product
                st.markdown("#### Top Buyers")
                top_buyers = product_data['User_id'].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_buyers.values, y=top_buyers.index, ax=ax)
                ax.set_title(f"Top 10 Buyers of '{selected_product}'")
                ax.set_xlabel("Number of Purchases")
                ax.set_ylabel("User ID")
                st.pyplot(fig)
                
                # Frequently co-purchased products
                st.markdown("#### Frequently Co-Purchased Products")
                
                # Find users who bought the selected product
                users_who_bought = product_data['User_id'].unique()
                
                # Find what else these users bought
                other_purchases = st.session_state.train_data[
                    (st.session_state.train_data['User_id'].isin(users_who_bought)) & 
                    (st.session_state.train_data['itemDescription'] != selected_product)
                ]
                
                if len(other_purchases) > 0:
                    co_purchased = other_purchases['itemDescription'].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=co_purchased.values, y=co_purchased.index, ax=ax)
                    ax.set_title(f"Top 10 Products Co-Purchased with '{selected_product}'")
                    ax.set_xlabel("Number of Purchases")
                    st.pyplot(fig)
                else:
                    st.info("No co-purchased products found.")
                    
                # Association rules involving this product
                st.markdown("#### Association Rules")
                
                # Initialize pattern miner if it hasn't been run yet
                if st.session_state.pattern_miner is None:
                    with st.spinner("Initializing pattern miner to find association rules..."):
                        st.session_state.pattern_miner = PatternMiner(st.session_state.train_data)
                        
                # Run pattern mining if it hasn't been run yet
                if not hasattr(st.session_state.pattern_miner, 'association_rules') or st.session_state.pattern_miner.association_rules is None:
                    with st.spinner("Running pattern mining to find association rules..."):
                        st.session_state.pattern_miner.run(algorithm='fpgrowth', min_support=0.01, min_confidence=0.5)
                
                # Display rules involving the selected product
                if hasattr(st.session_state.pattern_miner, 'association_rules') and st.session_state.pattern_miner.association_rules is not None and len(st.session_state.pattern_miner.association_rules) > 0:
                    # Convert frozensets to strings for display
                    rule_df = st.session_state.pattern_miner.association_rules.copy()
                    
                    # Convert antecedents and consequents to strings
                    rule_df['antecedents_str'] = rule_df['antecedents'].apply(lambda x: ', '.join(list(x)) if x else '')
                    rule_df['consequents_str'] = rule_df['consequents'].apply(lambda x: ', '.join(list(x)) if x else '')
                    
                    # Filter rules involving the selected product
                    product_rules = rule_df[
                        rule_df['antecedents_str'].str.contains(selected_product, regex=False) | 
                        rule_df['consequents_str'].str.contains(selected_product, regex=False)
                    ]
                    
                    if len(product_rules) > 0:
                        st.write(f"Association rules involving '{selected_product}':")
                        
                        # Display only relevant columns and rename for clarity
                        display_rules = product_rules[['antecedents_str', 'consequents_str', 'confidence', 'lift']].copy()
                        display_rules.columns = ['If purchased', 'Then likely to purchase', 'Confidence', 'Lift']
                        
                        st.dataframe(display_rules)
                        
                        # Create download button for rules
                        st.markdown("#### Download Association Rules")
                        create_download_button(
                            display_rules,
                            f"association_rules_{selected_product.replace(' ', '_')}.csv",
                            "Download Association Rules"
                        )
                    else:
                        st.info(f"No association rules found involving '{selected_product}'.")
                else:
                    min_support = st.session_state.pattern_miner.min_support if hasattr(st.session_state.pattern_miner, 'min_support') else "default"
                    st.info(f"No association rules were generated with the current settings (min_support={min_support}).")
                    st.write("Try lowering the minimum support threshold in the Pattern Mining section.")
            else:
                st.warning(f"No data found for product '{selected_product}'")
                
                # Add download button for product analysis report
                product_report = product_data[['User_id', 'itemDescription', 'Date']].copy() if 'Date' in product_data.columns else product_data[['User_id', 'itemDescription']].copy()
                
                if len(product_report) > 0:
                    st.markdown("#### Download Product Analysis")
                    create_download_button(
                        product_report,
                        f"product_{selected_product.replace(' ', '_')}_analysis.csv",
                        "Download Product Analysis"
                    )

    with analysis_tabs[2]:
        st.markdown("### Recommendation Comparison")
        
        st.markdown("#### Compare Different Recommendation Methods")
        
        # User selection for recommendations
        st.markdown("##### Select User for Recommendations")
        
        # Option to select a user
        user_selection = st.radio(
            "How would you like to select a user?",
            ["Choose from list", "Enter user ID", "Random user"],
            index=2,
            key="recommendation_user_selection"
        )
        
        selected_user = None
        
        if user_selection == "Choose from list":
            # Get a manageable list of users (top users by purchase count)
            top_users = st.session_state.train_data['User_id'].value_counts().head(100).index.tolist()
            selected_user = st.selectbox("Select a user", options=top_users, key="recommendation_user_dropdown")
            
        elif user_selection == "Enter user ID":
            user_id_input = st.text_input("Enter user ID", key="recommendation_user_input")
            if user_id_input:
                try:
                    user_id_input = float(user_id_input)
                    if user_id_input in st.session_state.train_data['User_id'].values:
                        selected_user = user_id_input
                    else:
                        st.warning(f"User ID {user_id_input} not found in the dataset")
                except ValueError:
                    st.warning("Please enter a valid user ID")
                    
        elif user_selection == "Random user":
            if st.button("Get Random User", key="recommendation_random_user"):
                selected_user = random.choice(st.session_state.train_data['User_id'].unique())
        
        # Generate recommendations if a user is selected
        if selected_user is not None:
            st.markdown(f"##### Recommendations for User {selected_user}")
            
            # Initialize components if needed
            if st.session_state.collaborative_filter is None:
                st.info("Initializing collaborative filter...")
                st.session_state.collaborative_filter = CollaborativeFilter(st.session_state.train_data)
                st.session_state.collaborative_filter.prepare_data()
            
            if st.session_state.pattern_miner is None:
                st.info("Initializing pattern miner...")
                st.session_state.pattern_miner = PatternMiner(st.session_state.train_data)
                st.session_state.pattern_min_support = 0.005
                st.session_state.pattern_miner.run(algorithm='fpgrowth', min_support=0.005, min_confidence=0.3)
            
            if st.session_state.integrated_system is None:
                st.info("Initializing integrated system...")
                st.session_state.integrated_system = IntegratedSystem(
                    st.session_state.train_data, 
                    st.session_state.test_data
                )
                st.session_state.integrated_system.mine_patterns(min_support=0.005, algorithm='fpgrowth')
                st.session_state.integrated_system.prepare_collaborative_filtering()
            
            # User's purchase history
            user_data = st.session_state.train_data[st.session_state.train_data['User_id'] == selected_user]
            user_items = set(user_data['itemDescription'].unique())
            
            st.markdown("##### User's Purchase History")
            st.write(f"User {selected_user} has purchased {len(user_items)} unique items:")
            
            # Display as a collapsible section
            with st.expander("View purchase history"):
                user_items_df = user_data['itemDescription'].value_counts().reset_index()
                user_items_df.columns = ['Item', 'Count']
                st.dataframe(user_items_df)
            
            # Generate recommendations using different methods
            st.markdown("##### Generate Recommendations")
            
            # Setup columns for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Collaborative Filtering Methods**")
                
                # Select CF method
                cf_method = st.radio(
                    "Select collaborative filtering method",
                    ["User-based", "Item-based", "SVD-based"],
                    index=2
                )
                
                method_map = {
                    "User-based": "user",
                    "Item-based": "item",
                    "SVD-based": "svd"
                }
                
                # Generate CF recommendations
                if st.button("Generate CF Recommendations"):
                    with st.spinner("Generating collaborative filtering recommendations..."):
                        cf_recommendations = st.session_state.collaborative_filter.generate_recommendations(
                            user_id=selected_user,
                            method=method_map[cf_method],
                            n_recommendations=10
                        )
                        
                        if cf_recommendations is not None and len(cf_recommendations) > 0:
                            st.write(f"{cf_method} recommendations:")
                            st.dataframe(cf_recommendations)
                        else:
                            st.warning(f"No {cf_method.lower()} recommendations could be generated for this user.")
            
            with col2:
                st.markdown("**Hybrid Recommendations**")
                
                # Generate hybrid recommendations
                if st.button("Generate Hybrid Recommendations"):
                    with st.spinner("Generating hybrid recommendations..."):
                        hybrid_recommendations = st.session_state.integrated_system.generate_recommendations(
                            user_id=selected_user,
                            n_recommendations=10,
                            with_patterns=True
                        )
                        
                        if hybrid_recommendations is not None and len(hybrid_recommendations) > 0:
                            st.write("Hybrid recommendations:")
                            st.dataframe(hybrid_recommendations)
                        else:
                            st.warning("No hybrid recommendations could be generated for this user.")
            
            # Compare recommendations if available
            if 'cf_recommendations' in locals() and 'hybrid_recommendations' in locals():
                st.markdown("##### Recommendation Comparison")
                
                # Check for overlap
                cf_items = set(cf_recommendations['itemDescription'])
                hybrid_items = set(hybrid_recommendations['itemDescription'])
                common_items = cf_items.intersection(hybrid_items)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CF Recommendations", len(cf_items))
                with col2:
                    st.metric("Hybrid Recommendations", len(hybrid_items))
                with col3:
                    st.metric("Common Items", len(common_items))
                
                # Visualize overlap with Venn diagram
                st.markdown("##### Recommendation Overlap")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    from matplotlib_venn import venn2
                    venn2([cf_items, hybrid_items], 
                        set_labels=[f"{cf_method} Recommendations", "Hybrid Recommendations"], 
                        ax=ax)
                    plt.title("Recommendation Method Comparison")
                    st.pyplot(fig)
                except ImportError:
                    st.error("matplotlib-venn library not installed. Cannot generate Venn diagram.")

    with analysis_tabs[3]:
        st.markdown("### Custom Queries")
        
        st.markdown("""
        This section allows you to run custom analysis queries on the dataset.
        """)
        
        # Example queries
        query_options = [
            "Top 20 most popular products",
            "Distribution of purchases per user",
            "Purchase patterns by day of week",
            "Product category analysis",
            "Custom SQL-like query"
        ]
        
        selected_query = st.selectbox("Select a query", options=query_options)
        
        if selected_query == "Top 20 most popular products":
            if st.button("Run Query"):
                with st.spinner("Analyzing product popularity..."):
                    # Get product counts
                    product_counts = st.session_state.train_data['itemDescription'].value_counts().reset_index()
                    product_counts.columns = ['Product', 'Purchase Count']
                    
                    # Display results
                    st.write("Top 20 most popular products:")
                    st.dataframe(product_counts.head(20))
                    
                    # Visualize
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(x='Purchase Count', y='Product', data=product_counts.head(20), ax=ax)
                    plt.title("Top 20 Most Popular Products")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Additional insights
                    total_purchases = len(st.session_state.train_data)
                    top20_purchases = product_counts.head(20)['Purchase Count'].sum()
                    percentage = (top20_purchases / total_purchases) * 100
                    
                    st.info(f"The top 20 products account for {percentage:.2f}% of all purchases.")
        
        elif selected_query == "Distribution of purchases per user":
            if st.button("Run Query"):
                with st.spinner("Analyzing user purchase distribution..."):
                    # Get purchase counts per user
                    user_counts = st.session_state.train_data['User_id'].value_counts().reset_index()
                    user_counts.columns = ['User_id', 'Purchase Count']
                    
                    # Basic statistics
                    mean_purchases = user_counts['Purchase Count'].mean()
                    median_purchases = user_counts['Purchase Count'].median()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Purchases per User", f"{mean_purchases:.2f}")
                    with col2:
                        st.metric("Median Purchases per User", f"{median_purchases:.2f}")
                    with col3:
                        st.metric("Total Users", len(user_counts))
                    
                    # Histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(user_counts['Purchase Count'], bins=50, kde=True, ax=ax)
                    plt.title("Distribution of Purchases per User")
                    plt.xlabel("Number of Purchases")
                    plt.ylabel("Number of Users")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # User segments
                    st.markdown("##### User Segments by Purchase Frequency")
                    
                    # Define segments
                    user_counts['Segment'] = pd.cut(
                        user_counts['Purchase Count'],
                        bins=[0, 5, 10, 20, 50, 100, float('inf')],
                        labels=['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
                    )
                    
                    # Count users in each segment
                    segment_counts = user_counts['Segment'].value_counts().sort_index()
                    
                    # Plot segments
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)
                    plt.title("User Segments by Purchase Frequency")
                    plt.xlabel("Number of Purchases")
                    plt.ylabel("Number of Users")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif selected_query == "Purchase patterns by day of week":
            if 'Date' in st.session_state.train_data.columns:
                if st.button("Run Query"):
                    with st.spinner("Analyzing purchase patterns by day of week..."):
                        # Make sure date is datetime
                        if not pd.api.types.is_datetime64_any_dtype(st.session_state.train_data['Date']):
                            date_data = pd.to_datetime(st.session_state.train_data['Date'], dayfirst=True)
                        else:
                            date_data = st.session_state.train_data['Date']
                        
                        # Extract day of week
                        day_of_week = date_data.dt.day_name()
                        
                        # Count purchases by day
                        day_counts = day_of_week.value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 
                                                                       'Thursday', 'Friday', 'Saturday', 'Sunday'])
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax)
                        plt.title("Purchase Patterns by Day of Week")
                        plt.xlabel("Day of Week")
                        plt.ylabel("Number of Purchases")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Find peak day
                        peak_day = day_counts.idxmax()
                        peak_count = day_counts.max()
                        
                        st.info(f"Peak shopping day is {peak_day} with {peak_count:,} purchases.")
            else:
                st.warning("Date column not found in the dataset. Cannot analyze purchase patterns by day of week.")
        
        elif selected_query == "Product category analysis":
            if st.button("Run Query"):
                with st.spinner("Analyzing product categories..."):
                    # Extract categories from product names
                    def extract_category(product_name):
                        if pd.isna(product_name):
                            return "Unknown"
                        
                        # Simple category extraction based on keywords
                        product_name = safe_str(product_name).lower()
                        
                        categories = {
                            "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
                            "meat": ["meat", "beef", "chicken", "pork", "steak", "sausage"],
                            "produce": ["fruit", "vegetable", "apple", "banana", "tomato", "lettuce", "fresh"],
                            "bakery": ["bread", "cake", "pastry", "bun", "roll"],
                            "beverages": ["water", "soda", "juice", "coffee", "tea", "drink"],
                            "snacks": ["chip", "crisp", "snack", "chocolate", "candy", "sweet"],
                            "canned": ["can", "canned", "soup", "preserved"],
                            "frozen": ["frozen", "ice cream", "freezer"],
                            "household": ["paper", "cleaner", "detergent", "soap", "household"]
                        }
                        
                        for category, keywords in categories.items():
                            for keyword in keywords:
                                if keyword in product_name:
                                    return category
                        
                        return "Other"
                    
                    # Apply category extraction
                    st.session_state.train_data['category'] = st.session_state.train_data['itemDescription'].apply(extract_category)
                    
                    # Count by category
                    category_counts = st.session_state.train_data['category'].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Purchase Count']
                    
                    # Display top categories
                    st.write("Top product categories:")
                    st.dataframe(category_counts.head(20))
                    
                    # Visualize top 10
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Purchase Count', y='Category', data=category_counts.head(10), ax=ax)
                    plt.title("Top 10 Product Categories")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Category diversity
                    st.metric("Total Categories", len(category_counts))
                    
                    # Remove the temporary category column
                    st.session_state.train_data = st.session_state.train_data.drop('category', axis=1)
        
        elif selected_query == "Custom SQL-like query":
            st.info("This feature allows you to run custom queries on the dataset.")
            
            # Example queries
            example_queries = [
                "Show top 10 users by purchase count",
                "Find products purchased by more than 100 users",
                "Count purchases by month (if date available)",
                "Custom query"
            ]
            
            query_type = st.selectbox("Select a query type", options=example_queries)
            
            if query_type == "Show top 10 users by purchase count":
                if st.button("Run Query"):
                    result = st.session_state.train_data['User_id'].value_counts().head(10).reset_index()
                    result.columns = ['User_id', 'Purchase Count']
                    st.dataframe(result)
            
            elif query_type == "Find products purchased by more than 100 users":
                if st.button("Run Query"):
                    product_users = st.session_state.train_data.groupby('itemDescription')['User_id'].nunique()
                    popular_products = product_users[product_users > 100].sort_values(ascending=False).reset_index()
                    popular_products.columns = ['Product', 'Number of Users']
                    st.dataframe(popular_products)
            
            elif query_type == "Count purchases by month (if date available)":
                if 'Date' in st.session_state.train_data.columns:
                    if st.button("Run Query"):
                        # Convert to datetime if needed
                        if not pd.api.types.is_datetime64_any_dtype(st.session_state.train_data['Date']):
                            date_data = pd.to_datetime(st.session_state.train_data['Date'], dayfirst=True)
                        else:
                            date_data = st.session_state.train_data['Date']
                        
                        # Extract month and count
                        monthly_counts = date_data.dt.to_period('M').value_counts().sort_index()
                        monthly_counts = pd.DataFrame({
                            'Month': monthly_counts.index.astype(str),
                            'Purchase Count': monthly_counts.values
                        })
                        
                        st.dataframe(monthly_counts)
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.lineplot(x='Month', y='Purchase Count', data=monthly_counts, ax=ax)
                        plt.title("Purchase Count by Month")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("Date column not found in the dataset.")
            
            elif query_type == "Custom query":
                st.write("Custom queries coming soon!")
            

# Run the app
if __name__ == "__main__":
    pass 