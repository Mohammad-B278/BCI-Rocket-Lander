# EEG-Based Motor Imagery Classification for BCI Control

* **Status:** ⚠️ Work in Progress
* **Author:** [Your Name/GitHub Username]
* **LinkedIn:** [Link to Your Profile]
* **GitHub Repository:** [Link to Your Repo]

---
## 1. Project Overview

This project is an exploration into the field of Brain-Computer Interfaces (BCIs). The primary goal is to develop a machine learning pipeline that can accurately classify different motor imagery tasks (e.g., imagining closing the left fist, right fist, or moving feet) from raw EEG data.

The current phase focuses on **offline analysis**—building, training, and rigorously evaluating models on a publicly available dataset. The ultimate vision is to adapt this trained model for **real-time application**, allowing a user wearing an EEG headset to control a simple rocket lander game I developed, purely through thought.


---

## 2. Technology Stack

This project leverages the power of Python's scientific computing and machine learning ecosystem.

* **Core Language:** Python
* **Data Handling & Computation:** NumPy
* **EEG Signal Processing:** MNE-Python
* **Machine Learning & Modeling:** Scikit-learn
* **Future Exploration:** TensorFlow / Keras for Deep Learning models

---

## 3. Project Structure

The repository is organized to separate data exploration, model training, and application logic.
.
├── bci_pipeline/
│   ├── model_iterations/      # Scripts for ML model training, evaluation, and optimisation
│   └── notebooks/             # Jupyter notebooks for data exploration & hyperparameter tuning
├── planning/
│   └── *.txt                  # Initial planning and architecture documents
├── rocket_lander_game/
│   ├── main.py                # Main entry point to run the game
│   ├── game.py                # Core game logic and state management
│   ├── rocket.py              # Rocket sprite class and physics
│   ├── landing_pad.py         # Landing pad sprite class
│   ├── settings.py            # Game configuration and constants
│   └── assets/                # Directory for game images and sprites
├── LICENSE
└── README.md
---

## 4. Methodology

The project follows a systematic pipeline, from raw data ingestion to model evaluation.

### 4.1. Dataset

The project utilizes the **EEG Motor Movement/Imagery Dataset** from PhysioNet. This is a well-known BCI dataset containing recordings from 109 subjects. Each recording was made using a 64-channel EEG system while subjects performed various motor imagery tasks.

### 4.2. Data Preprocessing

Raw EEG data is noisy and requires significant cleaning before it can be used for machine learning. The following steps were applied using the `MNE-Python` library:

1.  **Band-Pass Filtering:** A filter was applied to keep frequencies between **8 Hz and 35 Hz**. This isolates the mu (μ) and beta (β) brain rhythms, which are most relevant for motor imagery.
2.  **Notch Filtering:** A 50 Hz notch filter was used to remove electrical grid noise, a common source of interference in European datasets.
3.  **Epoching:** The continuous EEG signal was segmented into discrete trials (epochs), from **-1 to +4 seconds** around the stimulus cue. This captures the brain's preparatory activity before and during the motor imagery task.

### 4.3. Feature Extraction & Modeling

Several approaches were tested to find the most effective features for classification.

* **Common Spatial Patterns (CSP):** A technique to find spatial filters that maximize the variance between two classes of motor imagery signals. It's a highly effective feature extraction method for EEG data.
* **Riemannian Geometry:** An alternative approach that treats the covariance matrices of EEG signals as points on a Riemannian manifold, using their geometric properties for classification.
* **Classifier:** A **Support Vector Machine (SVM)** was used as the final classifier due to its robustness with high-dimensional data.

To prevent overfitting and find the best model configuration, **GridSearchCV** with **K-Fold Cross-Validation** was used for hyperparameter tuning. The final model was deliberately trained on a subset of **20 subjects** to ensure it generalizes well to unseen data rather than memorizing the training set.

---

## 5. Results & Findings

Multiple experiments were conducted to compare the effectiveness of different pipelines. The mean classification accuracy was used as the primary evaluation metric across all available subjects.

| Version | Model Configuration | Mean Accuracy | Standard Deviation |
| :--- | :--- | :--- | :--- |
| 1 | Baseline | 0.3951 | 0.2105 |
| 2 | **Optimised Hyperparams - CSP + SVM** | **0.6644** | **0.0142** |
| 3 | Non-optimised Hyperparams - Riemann + SVM | 0.5762 | 0.1249 |
| 4 | Optimised Hyperparams - Riemann + SVM | 0.5646 | 0.1257 |
| 5 | Optimised Hyperparams V2 - Riemann + SVM | 0.5825 | 0.1273 |
| 6 | Optimised Hyperparams (20 Subjects) - CSP + SVM | 0.6630 | 0.0156 |

**Key Finding:** The **CSP + SVM** approach with tuned hyperparameters provided the highest and most stable performance, achieving a **mean accuracy of ~66.4%**. The performance plateaued around this mark, suggesting that more complex, non-linear models may be required to capture deeper patterns in the data.

---

## 6. Future Work

This project is an ongoing effort with a clear roadmap for future development.

1.  **Integrate Deep Learning Models:** To push past the current performance plateau, the next step is to implement a **Convolutional Neural Network (CNN)** or a hybrid CNN-RNN model using TensorFlow. These models are capable of learning hierarchical features directly from the raw EEG signals.
2.  **Real-Time BCI Implementation:** The ultimate goal is to transition from offline analysis to a live system. This involves:
    * Connecting to a consumer-grade EEG headset (e.g., Muse, OpenBCI).
    * Building a real-time data processing pipeline.
    * Integrating the trained model's output to control the **rocket lander game**, providing live feedback to the user.
