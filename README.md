# IST-AAut

This repository contains lab materials for the IST-AAut (Machine Learning) course.

- Check [Report.pdf](Submissions/Report.pdf) for the methodology and analysis, with comparisons of models across different metrics.

## Content

### [Part 1: Regression with Synthetic Data](AAutLab2425.pdf)

This section covers regression analysis tasks using synthetic data to explore key concepts in predictive modeling.

- **ML-Submission1**: Contains the Jupyter Notebook for **Multiple Linear Regression with Outliers**, implementing outlier removal, cross-validation, and tuning techniques as described in the report.

- **ML-Submission2**: Contains the Jupyter Notebook for the **ARX Model**, focusing on time-series data and system response modeling, with parameter optimization techniques.

#### Learning Objectives
- Understand multiple linear regression with synthetic data containing noise and outliers.
- Apply ARX (Auto-Regressive with eXogenous input) models for time-series data analysis.
- Evaluate model robustness with cross-validation and tuning techniques.

#### Technologies
- Python (3.11) with `scikit-learn` and `statsmodels`.
- MATLAB® for additional analysis and model validation.

### [Part 2: Image Analysis - Martian Crater Detection](AAutLab2425.pdf)

This section explores image classification and segmentation tasks focused on low-resolution (48x48) Martian crater analysis.

- **ML-Submission3**: Contains the Jupyter Notebook for **Image Classification** using SVC and CNN models, with techniques for handling imbalanced data and data augmentation.

- **ML-Submission4**: Contains the Jupyter Notebook for **Image Segmentation**, implementing MLP-Fusion and U-Net models for pixel-wise segmentation.

#### Learning Objectives
- Develop machine learning models to classify crater vs. non-crater images.
- Apply segmentation techniques (patch-based and pixel-based) to delineate crater boundaries.
- Address data imbalance with techniques like SMOTE and data augmentation.

#### Technologies
- Python (3.11) with `torch`, `torchvision`, `torchmetrics`, and `pytorch-lightning`.
- [Optuna](https://optuna.org/) for hyperparameter tuning.

## Authors

- [João Gonçalves - sqrt(-1)](https://github.com/eusouojoao)
- [Teresa Nogueira - 13A!](https://github.com/FrolickingAsteroid)

## License

This work is licensed under a [Creative Commons Attribution Non Commercial Share Alike 4.0 International][cc-by-nc-sa].

[cc-by-nc-sa]: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
