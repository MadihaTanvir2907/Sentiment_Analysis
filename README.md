# 📊 Social Media Ecosystems - Twitter Sentiment Analysis Project

This project is part of the course **Social Media Ecosystems (4ME304): Fall Term 2022** at **Linnaeus University**. The objective was to analyze and process real-world Twitter data using Python, with a focus on Big Data Analytics (BDA) techniques, data cleaning, clustering, and visualization.

---

## 📌 Project Overview

The assignment involved collecting, processing, and analyzing Twitter data related to a specific event or topic using the Twitter API. Key Python libraries used include `tweepy`, `nltk`, `scikit-learn`, and others to conduct sentiment analysis and unsupervised clustering.

---

## ⚙️ Features and Workflow

### 🐦 1. Data Collection
- Created a Twitter Developer Account.
- Collected **1000+ tweets** on a selected topic using the `Tweepy` library and Twitter API.

### 🧹 2. Data Cleaning & Preprocessing
- Used the `NLTK` library for:
  - Tokenization
  - Stopword removal
  - Word stemming and normalization

### 📐 3. Text Feature Extraction
- Implemented **TF-IDF Vectorization** using `scikit-learn` to transform text into meaningful numerical features.

### 🔍 4. Clustering
- Applied **K-Means Clustering** to group tweets into 10 clusters.
- Evaluated clusters using the **Elbow Method**.
- Visualized clustering results with dimensionality reduction techniques and custom plotting.

---

## 📊 Visualization
- Plotted data using Python’s `Matplotlib` and `Seaborn`.

---

## 🧠 Theoretical Insights

A theoretical report (max 800 words) was developed based on key concepts from Big Data Analytics such as:
- 📈 Data as a Valuable Resource  
- 🔐 Data Privacy  
- 👥 User-Centric Data  
- 📊 Social Data Analytics  
- 🧠 Influence Analytics  

The report linked the practical outcomes of the project to academic perspectives, highlighting implications for business, productivity, and digital ecosystems.

---

## 🧰 Technologies & Libraries Used

| Category            | Tools & Libraries                              |
|---------------------|------------------------------------------------|
| **Programming**     | Python 3.x                                      |
| **Data Collection** | Tweepy                                          |
| **Preprocessing**   | NLTK, Pandas                                    |
| **Clustering**      | Scikit-learn (TF-IDF, KMeans)                   |
| **Visualization**   | Matplotlib, Seaborn, Looker Studio              |
| **IDE**             | Jupyter Notebook                                |
| **Version Control** | Git, GitHub                                     |

---

## 📁 Repository Structure
📦 twitter-sentiment-analysis/
┣ 📂 data/ # Raw and cleaned tweet datasets
┣ 📂 notebooks/ # Jupyter Notebooks for each analysis step
┣ 📂 visuals/ # Charts, cluster graphs, and word clouds
┣ 📄 report.pdf # Final theoretical & technical report
┣ 📄 README.md # Project documentation
┗ 📄 requirements.txt # Python dependencies

---

## 📬 Submission Info

- **Course**: 4ME304 - Social Media Ecosystems  
- **University**: Linnaeus University  
- **Deadline**: 08/12/2023  
- **Supervisors**:  
  - Ahmed Taiye Mohammed – [ahmedtaiye.mohammed@lnu.se](mailto:ahmedtaiye.mohammed@lnu.se)  
  - Alisa Lincke – [alisa.lincke@lnu.se](mailto:alisa.lincke@lnu.se)

---

## ✅ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
2. **Install dependencies **
pip install -r requirements.txt
3. **Launch the analysis**
   jupyter notebook notebooks/main_analysis.ipynb
4. 📝 License
This project is intended for educational use under Linnaeus University academic policy.

🙋‍♀️ Contact
For questions or contributions:

Madiha Tanvir
📧 mt223rg@student.lnu.se









