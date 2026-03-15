# Iris Classifier

A machine learning project for classifying iris flowers using scikit-learn, featuring model comparison, learning curves, and comprehensive evaluation metrics.

## Overview

This project implements a multi-class classification pipeline on the classic Iris dataset. It includes data preprocessing, model training, evaluation, and comparison of multiple machine learning algorithms.

## Features

- **Multiple ML Models**: Random Forest, Logistic Regression, and Decision Tree classifiers
- **Data Preprocessing**: StandardScaler normalization for optimal performance
- **Model Evaluation**: Confusion matrices, classification reports, and F1 scores
- **Learning Curves**: Visual analysis of model performance vs. training size
- **Feature Importance**: Analysis of which features contribute most to predictions
- **Unit Tests**: Automated tests for data pipeline validation
- **Reproducibility**: Fixed random seeds for consistent results

## Quick Start

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/angelbech/iris-classifier.git
cd iris-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

**Train the main Random Forest model:**
```bash
python train.py
```

**Compare multiple models:**
```bash
python compare_models.py
```

**Generate learning curves:**
```bash
python learning_curve.py
```

**Run tests:**
```bash
python test_data.py
```

## Results

The project generates several outputs in the `outputs/` directory:

- `confusion_matrix.png` - Visual representation of prediction accuracy
- `feature_importance.png` - Bar chart showing feature contributions
- `learning_curve.png` - Training vs. validation performance
- `model_comparison.csv` - F1 scores for all models
- `train_log.txt` - Training hyperparameters and metrics

### Model Performance

| Model | F1 Score (Macro) |
|-------|------------------|
| Random Forest | ~0.96 |
| Logistic Regression | ~0.97 |
| Decision Tree | ~0.93 |

*Results may vary slightly due to train/test split randomness*

## Project Structure

```
iris-classifier/
├── train.py              # Main training script
├── compare_models.py     # Model comparison pipeline
├── learning_curve.py     # Learning curve generation
├── test_data.py         # Unit tests for data pipeline
├── requirements.txt     # Python dependencies
├── outputs/            # Generated results (created on first run)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── learning_curve.png
│   ├── model_comparison.csv
│   └── train_log.txt
└── README.md           # This file
```

## Technical Details

### Dataset
- **Source**: Scikit-learn's built-in Iris dataset
- **Samples**: 150 (50 per class)
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)

### Pipeline
1. Load Iris dataset
2. Split into 80% training / 20% test (stratified)
3. Standardize features using StandardScaler
4. Train classifier(s)
5. Evaluate on test set
6. Generate visualizations and metrics

### Hyperparameters
- **Random Forest**: 100 estimators, max depth 4
- **Decision Tree**: Max depth 4
- **Logistic Regression**: Max iterations 200
- **Random Seed**: 42 (for reproducibility)

## Dependencies

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

See `requirements.txt` for specific versions.

## Testing

The project includes unit tests to validate:
- Train/test split sizes (120/30 samples)
- StandardScaler zero-centering

Run tests with:
```bash
python test_data.py
```

## Future Enhancements

- [ ] Add cross-validation
- [ ] Implement hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Add more advanced models (SVM, XGBoost)
- [ ] Create interactive visualization dashboard
- [ ] Add ROC curves and precision-recall analysis
- [ ] Implement model persistence (save/load trained models)

## License

This project is open source.

## Author

**Angel**
- GitHub: [@angelbech](https://github.com/angelbech)
- Email: angelbechar@gmail.com

## Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- Powered by scikit-learn
