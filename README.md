# 🌸 Iris Flower Classification Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Technical Implementation](#technical-implementation)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

This project implements a machine learning solution for classifying Iris flowers into three species (Setosa, Versicolor, and Virginica) based on their sepal and petal measurements. The model achieves **100% accuracy** on the test set using the K-Nearest Neighbors (KNN) algorithm.

**Author**: Sonny B. KolliO
**Internship**: CodeAlpha Data Science Intern
**Project Type**: Supervised Machine Learning - Classification

## 📊 Dataset Description

The Iris dataset is one of the most famous datasets in pattern recognition literature. It contains 150 samples with 4 features each:

### Features:
1. **Sepal Length** (cm)
2. **Sepal Width** (cm)
3. **Petal Length** (cm)
4. **Petal Width** (cm)

### Target Classes:
- **0**: Iris Setosa
- **1**: Iris Versicolor
- **2**: Iris Virginica

**Dataset Statistics**:
- Total samples: 150
- Features per sample: 4
- Classes: 3 (50 samples each)
- No missing values

## ✨ Features

### ✅ Core Features
- **100% Accuracy Model**: Perfect classification on test data
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Data Preprocessing**: Proper scaling and train-test splitting
- **Model Persistence**: Save and load trained models
- **Interactive Predictions**: Real-time flower classification

### 📈 Visualizations Generated
1. **Pairplot**: Relationships between all feature pairs
2. **Boxplots**: Distribution of each feature by species
3. **Confusion Matrix**: Model performance visualization

### 🔧 Technical Features
- Clean, modular Python code
- No warnings or errors
- Reproducible results
- Well-documented functions
- Model serialization for deployment

## 🛠️ Technical Implementation

### Algorithms Used
- **Primary Algorithm**: K-Nearest Neighbors (KNN)
- **Distance Metric**: Euclidean distance
- **Optimal k**: 1 neighbor
- **Scaling**: StandardScaler (mean=0, variance=1)

### Data Splitting
- **Training set**: 70% (105 samples)
- **Testing set**: 30% (45 samples)
- **Random seed**: 0 (for reproducibility)
- **Stratified sampling**: Maintains class distribution

### Model Performance