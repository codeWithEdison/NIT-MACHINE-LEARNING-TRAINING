# Machine Learning Algorithms Guide

## Supervised Learning

### 1. Linear Regression
- **Definition**: Predicts numerical values by finding best-fit line through data points
- **Example**: House price prediction
  - Input: Size (1500 sq ft), Bedrooms (3), Location score (8/10)
  - Output: Predicted price ($350,000)
- **Real-world use**: Stock price prediction, sales forecasting

### 2. Logistic Regression
- **Definition**: Predicts binary outcomes (yes/no) with probability
- **Example**: Customer churn prediction
  - Input: Usage time (5 months), Bill amount ($100), Support calls (3)
  - Output: 75% chance of churning
- **Real-world use**: Email spam detection, disease diagnosis

### 3. Decision Trees
- **Definition**: Makes decisions through series of if-then rules
- **Example**: Loan approval
  - If income > $50,000 and credit score > 700: Approve
  - If income < $50,000 and debt ratio > 40%: Deny
- **Real-world use**: Customer service chatbots, medical diagnosis

### 4. Random Forest
- **Definition**: Combines multiple decision trees for robust predictions
- **Example**: Disease risk assessment
  - Tree 1: Checks diet
  - Tree 2: Analyzes exercise
  - Tree 3: Evaluates family history
  - Final prediction: Average of all trees
- **Real-world use**: Credit risk assessment, image classification

### 5. Support Vector Machine (SVM)
- **Definition**: Separates data categories with maximum margin
- **Example**: Image classification
  - Input: Image pixels
  - Output: Cat or dog classification
- **Real-world use**: Face detection, handwriting recognition

### 6. K-Nearest Neighbors (KNN)
- **Definition**: Classifies based on most similar known cases
- **Example**: Movie recommendations
  - Find 5 users with similar taste
  - Recommend movies they liked
- **Real-world use**: Product recommendations, pattern recognition

### 7. Naive Bayes
- **Definition**: Uses probability for quick classifications
- **Example**: Sentiment analysis
  - Input: "Great product, love it!"
  - Output: Positive sentiment (90% confidence)
- **Real-world use**: Spam filtering, document categorization

## Unsupervised Learning

### 1. K-Means Clustering
- **Definition**: Groups similar items automatically
- **Example**: Customer segmentation
  - Group 1: High spenders, frequent purchases
  - Group 2: Moderate spenders, occasional purchases
  - Group 3: Low spenders, rare purchases
- **Real-world use**: Market segmentation, image compression

### 2. Principal Component Analysis (PCA)
- **Definition**: Reduces data complexity while preserving patterns
- **Example**: Image compression
  - Input: 1000x1000 pixel image
  - Output: Compressed version with key features
- **Real-world use**: Facial recognition, dimensionality reduction

### 3. Hierarchical Clustering
- **Definition**: Creates nested groups of similar items
- **Example**: Document organization
  - Level 1: Technology, Sports, Politics
  - Level 2: Mobile, Computers, Software
  - Level 3: Specific topics
- **Real-world use**: Biological taxonomy, document organization

## Neural Networks

### 1. Feedforward Neural Networks
- **Definition**: Basic neural network for pattern recognition
- **Example**: Digit recognition
  - Input: Handwritten digit image
  - Output: Number classification (0-9)
- **Real-world use**: Speech recognition, quality control

### 2. Convolutional Neural Networks (CNNs)
- **Definition**: Specialized networks for image analysis
- **Example**: Object detection
  - Input: Street photo
  - Output: Identified cars, pedestrians, signs
- **Real-world use**: Medical image analysis, facial recognition

### 3. Recurrent Neural Networks (RNNs)
- **Definition**: Networks that process sequences with memory
- **Example**: Language translation
  - Input: English sentence
  - Output: Spanish translation
- **Real-world use**: Speech recognition, text generation

## Reinforcement Learning

### 1. Q-Learning
- **Definition**: Learns optimal actions through rewards
- **Example**: Game AI
  - State: Current game position
  - Action: Move selection
  - Reward: Points scored
- **Real-world use**: Robot navigation, game AI

### 2. Deep Q-Networks (DQN)
- **Definition**: Combines Q-learning with neural networks
- **Example**: Self-driving car
  - Input: Camera and sensor data
  - Output: Steering, acceleration decisions
- **Real-world use**: Robotics control, game AI

## Practical Tips
1. Start with simpler algorithms (Linear/Logistic Regression)
2. Match algorithm to problem type:
   - Prediction → Supervised Learning
   - Grouping → Clustering
   - Image/Video → CNNs
   - Sequential data → RNNs
3. Consider computational resources and data size
4. Evaluate multiple algorithms for best results
