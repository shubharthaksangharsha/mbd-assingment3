# Mining Big Data - Assignment 3

## Overview
This project implements a comprehensive recommendation system for a grocery store, combining pattern mining and collaborative filtering techniques. It consists of four main components:

1. **Task A: Pattern Mining** - Implementing frequent itemset mining using Apriori and FP-Growth algorithms
2. **Task B: Collaborative Filtering** - Building a recommendation system using user-based, item-based, and SVD approaches
3. **Task C: Integration** - Integrating pattern mining and collaborative filtering for enhanced recommendations
4. **Task D: Interactive Analysis** - Tools for in-depth analysis of purchase patterns and recommendation comparison

## Project Structure
```
├── dataset/                  # Dataset files
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test dataset
├── src/                      # Source code
│   ├── task_a/               # Pattern mining implementation
│   │   └── pattern_mining.py # Pattern mining algorithms
│   ├── task_b/               # Collaborative filtering implementation
│   │   └── collaborative_filtering.py  # CF algorithms
│   └── task_c/               # Integrated system
│       └── integration.py    # Hybrid recommendation system
├── utils/                    # Utility functions
│   └── data_utils.py         # Data preprocessing utilities
├── visualizations/           # Generated visualizations
├── results/                  # Generated results and metrics
├── main.py                   # Interactive main application
├── app.py                    # Streamlit interactive web application
├── run_task_a.py             # Script to run Task A independently
├── run_task_b.py             # Script to run Task B independently
├── run_task_c.py             # Script to run Task C independently
├── findings_task_a.txt       # Detailed findings from Task A
├── findings_task_b.txt       # Detailed findings from Task B
├── findings_task_c.txt       # Detailed findings from Task C
├── findings_task_app.txt     # Documentation of app debugging and fixes
├── requirements.txt          # Required packages
└── README.md                 # Project documentation
```

## Requirements
- Python 3.8 or higher
- Required packages as listed in `requirements.txt`:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - mlxtend
  - scikit-learn
  - networkx
  - matplotlib-venn
  - streamlit (v1.44.1+)
  - pyfpgrowth

## Installation
1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Running the Streamlit Web Application
```
streamlit run app.py
```
This launches an interactive web application with a user-friendly interface for all tasks.

### Running Individual Tasks
- **Task A (Pattern Mining)**:
  ```
  python run_task_a.py
  ```
- **Task B (Collaborative Filtering)**:
  ```
  python run_task_b.py
  ```
- **Task C (Integration)**:
  ```
  python run_task_c.py
  ```

## Tasks

### Task A: Pattern Mining
This task implements frequent itemset mining to discover patterns in customer purchase behavior.

**Features:**
- Support for both Apriori and FP-Growth algorithms
- Configurable minimum support and confidence thresholds
- Generation of association rules to identify product relationships
- Visualization of frequent itemsets and rules
- Export of results to CSV files

**Key Metrics:**
- Support: The frequency of an itemset in the dataset
- Confidence: The likelihood of item Y being purchased when item X is purchased
- Lift: The strength of association between items

### Task B: Collaborative Filtering
This task implements multiple collaborative filtering approaches to generate personalized recommendations.

**Features:**
- User-based collaborative filtering: Recommends items based on similar users
- Item-based collaborative filtering: Recommends items similar to those a user has purchased
- Matrix Factorization (SVD): Decomposes the user-item matrix to capture latent factors
- Method comparison with performance metrics
- Visualization of recommendations and similarity matrices

**Key Metrics:**
- Hit Rate: Proportion of recommended items that users actually purchase
- Precision: Accuracy of recommendations
- Coverage: Range of items that can be recommended
- Diversity: Variety in recommendations

### Task C: Integration
This task combines pattern mining and collaborative filtering to create a hybrid recommendation system.

**Features:**
- Weighted integration of collaborative filtering and pattern-based recommendations
- Boost for items that appear in both recommendation sets
- Comprehensive evaluation of hybrid vs. individual approaches
- Visualization of recommendation performance
- Detailed findings report (findings_task_c.txt)

**Key Findings:**
- Pattern-based approach achieved a hit rate of 0.1337 and precision of 0.2475
- Collaborative filtering achieved better coverage (0.4012) and diversity (0.4427)
- Hybrid approach provided a balanced middle ground with hit rate of 0.1137
- Pattern mining is effective for product associations while CF excels at personalization

### Task D: Interactive Analysis
This component provides specialized tools for deeper analysis of shopping patterns and recommendations.

**Features:**
- Analyze frequent items for specific users
- Find users with similar purchase patterns
- Discover complementary products using association rules
- Compare different recommendation methods side by side
- Generate personalized shopping baskets with category organization
- Export personalized shopping lists

## Web Application Features
The Streamlit-based web application offers a user-friendly interface for interacting with all components of the system:

1. **Navigation**: Simple tab-based navigation between tasks
2. **Data Exploration**: View and filter dataset statistics
3. **Interactive Visualization**: Dynamic charts and graphs for pattern analysis
4. **User Selection**: Multiple options for selecting users for recommendations
5. **Method Comparison**: Side-by-side comparison of recommendation approaches
6. **Export Options**: Download results in various formats
7. **Configuration**: Adjust parameters for each algorithm through intuitive controls

## Results and Outputs
The system generates several outputs:

- **Visualizations Directory:**
  - Frequent itemset visualizations
  - Association rule networks
  - Recommendation comparison charts
  - User similarity matrices
  - Purchase pattern visualizations

- **Results Directory:**
  - Frequent itemsets (CSV)
  - Association rules (CSV)
  - Recommendation metrics (CSV)
  - Personalized shopping lists (TXT)
  - Method comparison results (CSV)

- **Findings Reports:**
  - Detailed analysis of recommendation approaches
  - Performance metrics and comparisons
  - Business implications
  - Limitations and future improvements
  - App debugging documentation

## Tips for Optimal Use
1. **For Pattern Mining:**
   - Use FP-Growth for larger datasets (faster than Apriori)
   - Start with support = 0.01 and adjust based on results
   - Focus on rules with high lift values (>1.5)

2. **For Collaborative Filtering:**
   - SVD typically provides the best balance of quality and speed
   - User-based CF works well for users with rich purchase history
   - Item-based CF works better for new or infrequent users

3. **For Integrated System:**
   - The hybrid approach provides the most balanced recommendations
   - Adjust the weighting (default: 60% CF, 40% pattern-based) for different scenarios
   - Use the evaluation metrics to determine the best approach for specific use cases

4. **For Interactive Analysis:**
   - The shopping basket generator provides practical, categorized recommendations
   - The complementary products tool is excellent for cross-selling strategies
   - User similarity analysis helps identify market segments

5. **For Web Application:**
   - Allow visualizations to fully load before switching tabs
   - For large datasets, use the FP-Growth algorithm to improve performance
   - When comparing methods, start with a small number of users for faster evaluation

## Troubleshooting
- **User_id Column Issues**: If you encounter 'User_id' related errors, check that column naming is consistent across datasets. The system has built-in handling for variant spellings (user_id, userId, etc.)
- **Visualization Reset**: If visualizations reset when clicked, try using the "Regenerate Visualizations" button or check that Streamlit session state is being properly managed
- **Performance Issues**: 
  - For slow execution with large datasets, reduce the number of users in the evaluation (n_users parameter)
  - Use FP-Growth instead of Apriori for large transaction databases
  - Consider running individual tasks (run_task_*.py) instead of the full app for faster processing
- **Missing Data Warnings**: Large numbers of missing values are expected in the dataset and are handled automatically. Check the console for details on records removed
- **SettingWithCopyWarning**: These warnings from pandas are informational and don't affect functionality
- **Memory Issues**: If encountering memory errors, try closing other applications or reducing the dataset size
- **Association Rules Naming**: If encountering errors related to accessing rules, check the attribute name ('association_rules' vs 'rules')

For detailed information about debugging and fixes that were implemented, refer to the `findings_task_app.txt` file.

## Version Control
The project is version controlled using Git. Key information:
- Initial version: v1.0.0 (Base implementation of all tasks)
- Current version: v1.1.0 (Bug fixes and performance improvements)
- Release notes:
  - Fixed User_id column inconsistency issues
  - Resolved association rules naming problems
  - Optimized evaluation process to use fewer users by default
  - Added comprehensive error handling
  - Updated UI components to use latest Streamlit parameters
  - Added detailed debugging documentation

The repository has been initialized with Git, and all code changes have been committed. To clone this repository:
```bash
git clone <repository-url>
cd MBD_ASSIGNMENT3
```

## Authors
- Assignment for Mining Big Data, Master of AIML program, University of Adelaide 