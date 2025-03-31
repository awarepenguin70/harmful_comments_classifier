# 🚨 Harmful Comment Classification System  

## 📌 Overview  
This project aims to automatically **detect and classify harmful or offensive comments** on social media platforms using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. It compares two classification models—**Naïve Bayes (MultinomialNB)** and **k-Nearest Neighbors (k-NN)**—to determine which performs better in identifying toxic content.

---

## 🔍 Key Features  
✔️ **Text Preprocessing** – Cleans and normalizes raw text data  
✔️ **Feature Extraction** – Uses TF-IDF and Count Vectorization  
✔️ **Model Comparison** – Evaluates Naïve Bayes vs. k-NN  
✔️ **Performance Metrics** – Accuracy, Precision, Recall, F1-Score  
✔️ **Visual Insights** – Confusion Matrix, ROC Curves, Word Clouds  

---

## 🛠️ Installation & Setup  
### Prerequisites  
- Python 3.8+  
- Required packages in `requirements.txt`  

---

## 📊 Algorithms & How They Work  

### 1️⃣ Text Preprocessing  
Before feeding text into ML models, the following steps are applied:  
- **Lowercasing:** Converts text to lowercase (e.g., "You're STUPID" → "you're stupid")  
- **Removing URLs & Special Characters:** Eliminates irrelevant symbols  
- **Stopword Removal:** Removes common words like "the," "and," "is"  
- **Tokenization & Stemming:** Converts words to base forms (e.g., "running" → "run")  

---

### 2️⃣ Feature Extraction  
Text is converted into numerical features using:  
- **TF-IDF (Term Frequency-Inverse Document Frequency)**  
   - Weighs words by importance in a document vs. corpus  
   - Identifies rare but significant toxic terms (e.g., "kill," "worthless")  

- **Count Vectorization**  
   - Simple word frequency count  
   - Useful for detecting common toxic phrases  

---

### 3️⃣ Machine Learning Models  

#### 🔹 Naïve Bayes (MultinomialNB)  
- **How it works:**  
   - Uses Bayes' Theorem to predict class probabilities  
   - Treats each word as an independent feature (bag-of-words)  

- ✅ **Pros:**  
   - Fast training & prediction  
   - Effective with high-dimensional text data  

- ❗️ **Cons:**  
   - Struggles with word dependencies (e.g., sarcasm)  

---

#### 🔹 k-Nearest Neighbors (k-NN)  
- **How it works:**  
   - Classifies comments based on similarity to nearest training examples  
   - Uses Euclidean distance in vector space  

- ✅ **Pros:**  
   - No training required (lazy learning)  
   - Adapts to new patterns in data  

- ❗️ **Cons:**  
   - Slower at prediction time  
   - Sensitive to imbalanced datasets  

---

## 📈 Performance Comparison  

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Naïve Bayes (TF-IDF)    | 86.4%    | 100%      | 66.7%  | 80.0%    |
| k-NN (TF-IDF)           | 95.5%    | 90%       | 100%   | 94.7%    |
| Naïve Bayes (Count)     | 95.5%    | 100%      | 88.9%  | 94.1%    |
| k-NN (Count)            | 63.6%    | 100%      | 11.1%  | 20.0%    |

---

## 🎯 Key Takeaways  
✅ **Best Model:** Naïve Bayes with Count Vectorization (95.5% accuracy, 94.1% F1-Score)  
✅ **Best for Precision:** Naïve Bayes (100% precision) – No false positives  
✅ **Best for Recall:** k-NN with TF-IDF (100% recall) – Catches all harmful comments  

---

## 🚀 Future Improvements  
🔸 **Add Deep Learning:** Implement BERT, LSTM for better contextual understanding  
🔸 **Expand Dataset:** Include sarcasm, slang, and multilingual comments  
🔸 **Real-Time API:** Build an API for live comment moderation  
🔸 **Bias Mitigation:** Reduce false positives for minority dialects  

---

## 📜 License  
This project is open-source under the **MIT License**.  

---

## 📬 Contact  
For questions or contributions, reach out:  
📧 **Email:** [rahul004prasad@gmail.com](mailto:rahul004prasad@gmail.com)  
🌐 **GitHub:** [awarepenguin70](https://github.com/awarepenguin70)  

---

## 📝 Conclusion  
This system provides a strong baseline for harmful comment detection and can be deployed in moderation pipelines. While Naïve Bayes performs best, future work should focus on improving recall for nuanced toxicity.
