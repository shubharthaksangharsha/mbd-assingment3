# 📦 Mining Big Data - Assignment 3

### 🔗 [👉 Launch Web Application](https://mbd-assignment3.duckdns.org)

## 🧠 Overview
This project implements a comprehensive **recommendation system** for a grocery store by combining **pattern mining** and **collaborative filtering** techniques. It features:

1. 🧾 **Task A: Pattern Mining** - Apriori & FP-Growth algorithms  
2. 👥 **Task B: Collaborative Filtering** - User-based, Item-based & SVD methods  
3. 🔗 **Task C: Integration** - Hybrid recommendation system  
4. 🧪 **Task D: Interactive Analysis** - Dynamic tools for pattern analysis & recommendation comparison

---

## 📁 Project Structure
```text
├── dataset/                  
│   ├── train.csv             
│   └── test.csv              
├── src/                      
│   ├── task_a/               
│   │   └── pattern_mining.py 
│   ├── task_b/               
│   │   └── collaborative_filtering.py  
│   └── task_c/               
│       └── integration.py    
├── utils/                    
│   └── data_utils.py         
├── visualizations/           
├── results/                  
├── main.py                   
├── app.py                    # Streamlit Web App
├── run_task_a.py             
├── run_task_b.py             
├── run_task_c.py             
├── findings_task_a.txt       
├── findings_task_b.txt       
├── findings_task_c.txt       
├── findings_task_app.txt     
├── requirements.txt          
└── README.md                 
```
---

## ⚙️ Task A: Pattern Mining

This task identifies frequent itemsets using two algorithms:

- **Apriori Algorithm**: Classic bottom-up approach exploring frequent itemsets.
- **FP-Growth Algorithm**: More efficient by constructing a compact prefix tree.

### 📄 Output Files
- `findings_task_a.txt`: Contains top frequent itemsets, their supports, and comparison.
- `results/`: Stores mined patterns and intermediate results.

---

## 🤝 Task B: Collaborative Filtering

Implements recommendation models based on user interactions:

- **User-Based Filtering**: Finds similar users and recommends what they like.
- **Item-Based Filtering**: Suggests items similar to those already liked.
- **SVD (Matrix Factorization)**: Learns latent features to predict user-item interactions.

### 📄 Output Files
- `findings_task_b.txt`: Evaluation results (e.g., precision, recall) and examples of recommendations.

---

## 🔗 Task C: Integration

Combines pattern mining with collaborative filtering to build a hybrid recommender:

- Merges frequent itemsets with predicted ratings.
- Adjusts recommendations using item popularity and co-occurrence strength.

### 📄 Output Files
- `findings_task_c.txt`: Integration strategy, performance improvements, and hybrid analysis.

---

## 🧪 Task D: Interactive Web Application

A Streamlit-based dashboard that enables:

- Pattern exploration through dynamic visualizations
- Interactive comparison of recommendation methods
- Real-time generation of personalized recommendations

### 📄 Output Files
- `findings_task_app.txt`: Observations on usability, performance, and demo results.

---

## 🚀 Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run tasks separately
python run_task_a.py
python run_task_b.py
python run_task_c.py

# Launch the web application
streamlit run app.py
```
---

## 🧩 Dependencies

This project relies on the following Python libraries:

```txt
pandas
numpy
scikit-learn
mlxtend
surprise
streamlit
matplotlib
```
---

## 📊 Evaluation Metrics

The performance of the recommendation models is assessed using:

- **Precision@K**: Measures how many of the top-K recommended items are relevant.
- **Recall@K**: Evaluates how many of the relevant items are captured in the top-K.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **Support & Confidence**: Used in pattern mining to evaluate the strength of itemsets and rules.

---

## ✍️ Authors

- **Shubharthak Sangharsha** - [@shubharthak](https://shubharthaksangharsha.github.io/)

Feel free to add contributors or collaborators here.

---

## 📜 License

This project is part of the **Mining Big Data** course at the **University of Adelaide** and is intended for academic use.

---

## 🙌 Acknowledgements

Special thanks to the developers of:

- [`mlxtend`](http://rasbt.github.io/mlxtend/) - For implementing Apriori and FP-Growth.
- [`Surprise`](https://surpriselib.com/) - For collaborative filtering and SVD.
- [`Streamlit`](https://streamlit.io/) - For building interactive dashboards effortlessly.

---





