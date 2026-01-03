
# 🐾 Animal Image Classification Project

This project explores **machine learning approaches for animal image classification**, focusing on three categories: **Cats, Dogs, and Snakes**. The goal is to evaluate and compare different algorithms in terms of accuracy, precision, recall, and F1-score, while analyzing their strengths and limitations.

## 📂 Dataset
- Images of **cats, dogs, and snakes** collected under diverse conditions (lighting, backgrounds, poses).  
- Preprocessed by resizing to **64×64 pixels** and flattening into **1D feature vectors**.  
- Designed to test both simple and advanced classification models.

## ⚙️ Models Implemented
1. **Naive Bayes Classifier**  
   - Fast and lightweight.  
   - Struggles with high-dimensional image data due to independence assumptions.  

2. **Decision Tree Classifier**  
   - Easy to interpret and visualize.  
   - Performs slightly better than Naive Bayes but prone to overfitting.  

3. **Feedforward Neural Network (2 layers)**  
   - Learns complex pixel interactions.  
   - Achieved the **best performance** among the three models.  

## 📊 Results
| Model                     | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| Naive Bayes                | 41.17%   | 41.64%    | 41.17% | 41.32%   |
| Decision Tree              | 43.17%   | 42.96%    | 43.17% | 43.04%   |
| Feedforward Neural Network | **53.17%** | **54.58%** | **53.17%** | **51.78%** |

- **Best model:** Feedforward Neural Network  
- **Weakest model:** Naive Bayes (due to feature independence assumption)  
- **Key challenge:** Misclassification between visually similar classes (especially Cats vs Dogs).  
 

## 🎯 Conclusion
This project demonstrates how different machine learning models handle image classification tasks. While simple models like Naive Bayes and Decision Trees provide baseline results, neural networks show clear advantages in capturing complex visual patterns. With more data and advanced architectures, performance can be significantly improved.

