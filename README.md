# Harmful Comment Classification System

## Overview
This project develops an automated system to detect and classify harmful or offensive comments on social media platforms using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system compares two classification models—Naïve Bayes (MultinomialNB) and k-Nearest Neighbors (k-NN)—to determine optimal performance for identifying toxic content.

---

## Key Features
- **Text Preprocessing**: Comprehensive cleaning and normalization of raw text data
- **Feature Extraction**: Implementation of TF-IDF and Count Vectorization techniques
- **Model Comparison**: Systematic evaluation of Naïve Bayes versus k-NN algorithms
- **Performance Metrics**: Comprehensive analysis using accuracy, precision, recall, and F1-score
- **Visual Analytics**: Confusion matrices, ROC curves, and word cloud visualizations

---

## Installation & Setup
### Prerequisites
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Installation Steps
```bash
pip install -r requirements.txt
```

---

## Methodology

### Text Preprocessing Pipeline
The system applies a multi-stage preprocessing approach to prepare text data for machine learning:

1. **Text Normalization**
   - Converts all text to lowercase for consistency
   - Example: "You're STUPID" → "you're stupid"

2. **Content Cleaning**
   - Removes URLs, special characters, and irrelevant symbols
   - Preserves meaningful punctuation for context

3. **Stopword Removal**
   - Eliminates common words ("the," "and," "is") that don't contribute to toxicity detection
   - Maintains domain-specific terms that may indicate harmful content

4. **Tokenization and Stemming**
   - Breaks text into individual tokens
   - Reduces words to their base forms (e.g., "running" → "run")

### Feature Extraction Methods

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Purpose**: Weighs words by their importance within individual documents relative to the entire corpus
- **Advantage**: Effectively identifies rare but significant toxic terms (e.g., "kill," "worthless")
- **Application**: Particularly effective for detecting nuanced harmful language

#### Count Vectorization
- **Purpose**: Represents text as simple word frequency counts
- **Advantage**: Efficient for detecting common toxic phrases and patterns
- **Application**: Useful for identifying frequently used offensive terminology

### Machine Learning Models

#### Naïve Bayes (MultinomialNB)
**Algorithm Overview**:
- Utilizes Bayes' Theorem to calculate class probabilities
- Treats each word as an independent feature (bag-of-words approach)
- Assumes independence between features for computational efficiency

**Strengths**:
- Fast training and prediction times
- Highly effective with high-dimensional text data
- Robust performance with limited training data

**Limitations**:
- Struggles with word dependencies and contextual relationships
- May miss subtle forms of toxicity like sarcasm or implied threats

#### k-Nearest Neighbors (k-NN)
**Algorithm Overview**:
- Classifies comments based on similarity to nearest training examples
- Uses Euclidean distance measurements in high-dimensional vector space
- Implements lazy learning approach (no explicit training phase)

**Strengths**:
- No training time required
- Adapts naturally to new patterns in data
- Preserves local structure in the feature space

**Limitations**:
- Computationally expensive during prediction
- Sensitive to imbalanced datasets
- Performance degrades with high-dimensional sparse data

---

## Performance Analysis

### Quantitative Results

| Model Configuration          | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| Naïve Bayes + TF-IDF        | 86.4%    | 100%      | 66.7%  | 80.0%    |
| k-NN + TF-IDF               | 95.5%    | 90%       | 100%   | 94.7%    |
| Naïve Bayes + Count Vector  | 95.5%    | 100%      | 88.9%  | 94.1%    |
| k-NN + Count Vector         | 63.6%    | 100%      | 11.1%  | 20.0%    |

### Key Findings

**Best Overall Performance**: Naïve Bayes with Count Vectorization
- Achieved 95.5% accuracy with 94.1% F1-score
- Balanced performance across all metrics
- Optimal for production deployment

**Highest Precision**: Naïve Bayes configurations
- Achieved 100% precision in multiple configurations
- Zero false positives, ensuring legitimate comments aren't flagged
- Critical for maintaining user trust

**Highest Recall**: k-NN with TF-IDF
- Achieved 100% recall, capturing all harmful comments
- Important for comprehensive content moderation
- May require additional filtering to reduce false positives

---

## Technical Implementation

### Model Training Process
1. **Data Preparation**: Text preprocessing and feature extraction
2. **Train-Test Split**: 80-20 split for model evaluation
3. **Cross-Validation**: 5-fold cross-validation for robust performance assessment
4. **Hyperparameter Tuning**: Grid search for optimal model parameters

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall

---

## Future Enhancements

### Advanced Deep Learning Integration
- **BERT Implementation**: Leverage pre-trained transformers for better contextual understanding
- **LSTM Networks**: Capture sequential dependencies in text
- **Ensemble Methods**: Combine multiple models for improved performance

### Dataset Expansion
- **Sarcasm Detection**: Include training data for implicit toxicity
- **Slang and Colloquialisms**: Expand vocabulary coverage
- **Multilingual Support**: Extend system to multiple languages
- **Temporal Adaptation**: Regular updates to handle evolving language patterns

### Production Deployment
- **Real-Time API**: Develop RESTful API for live comment moderation
- **Scalable Architecture**: Implement distributed processing capabilities
- **A/B Testing Framework**: Continuous model improvement through controlled experiments

### Bias Mitigation
- **Fairness Auditing**: Regular assessment of model bias across demographic groups
- **Dialect Sensitivity**: Reduce false positives for minority language variants
- **Cultural Context**: Incorporate cultural nuances in toxicity detection

---

## Usage Instructions

### Basic Implementation
```python
from harmful_comment_classifier import HarmfulCommentClassifier

# Initialize classifier
classifier = HarmfulCommentClassifier(model='naive_bayes', vectorizer='count')

# Train model
classifier.train(training_data)

# Predict toxicity
result = classifier.predict("Sample comment text")
```

### API Integration
```python
# REST API endpoint
POST /classify
{
    "text": "Comment to classify",
    "model": "naive_bayes",
    "threshold": 0.5
}
```

---

## Technical Specifications

### System Requirements
- **Memory**: Minimum 8GB RAM for large-scale processing
- **Storage**: 2GB for model artifacts and training data
- **Processing**: Multi-core CPU recommended for batch processing

### Dependencies
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+
- nltk 3.7+
- matplotlib 3.5+
- seaborn 0.11+

---

## Research Implications

This system provides a robust foundation for automated content moderation with several key contributions:

1. **Comparative Analysis**: Systematic evaluation of traditional ML approaches for toxicity detection
2. **Feature Engineering**: Demonstration of effective text preprocessing techniques
3. **Performance Benchmarking**: Comprehensive metrics for model selection
4. **Practical Implementation**: Production-ready system architecture

The results indicate that while Naïve Bayes offers the best balance of performance and efficiency, the choice of model should depend on specific use case requirements regarding precision versus recall.

---

## License
This project is distributed under the MIT License. See LICENSE file for details.

---

## Contact Information
For questions, contributions, or collaboration opportunities:

**Email**: rahul004prasad@gmail.com  
**GitHub**: [awarepenguin70](https://github.com/awarepenguin70)  


---

## Conclusion

This harmful comment classification system demonstrates effective application of traditional machine learning techniques to content moderation challenges. The comparative analysis reveals that Naïve Bayes with Count Vectorization provides optimal performance for most use cases, achieving 95.5% accuracy with strong precision-recall balance.

The system's modular architecture enables easy integration into existing moderation pipelines while providing flexibility for future enhancements. As online discourse continues to evolve, this foundation supports ongoing development of more sophisticated toxicity detection capabilities.

Future work should prioritize improving recall for nuanced forms of toxicity while maintaining the system's strong precision performance. Integration of modern deep learning approaches presents opportunities for significant performance improvements, particularly in handling contextual and implicit forms of harmful content.
