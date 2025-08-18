EEG-Based Motor Imagery Classification for BCI Control
Status: ⚠️ Work in Progress

Author: Mohammad Beit Lafteh

LinkedIn: https://www.linkedin.com/in/mohammad-beit-lafteh/

1. Project Overview
This project is an exploration into the field of Brain-Computer Interfaces (BCIs). The primary goal is to develop a machine learning pipeline that can accurately classify three distinct motor imagery tasks (imagining closing the left fist, right fist, or moving both feet) from raw EEG data.

The current phase focuses on offline analysis—building, training, and rigorously evaluating models on a publicly available dataset. The ultimate vision is to adapt a high-performing model for real-time application, allowing a user to control external software, such as a simple game, purely through thought.

2. Technology Stack
This project leverages the power of Python's scientific computing and machine learning ecosystem.

Core Language: Python

Data Handling & Computation: NumPy

EEG Signal Processing: MNE-Python

Classical Machine Learning: Scikit-learn

Deep Learning (Planned): TensorFlow / Keras

3. Project Structure
The repository is organized to separate data exploration, model training, and application logic.

"""
.
├── bci_pipeline/
│   ├── model_iterations/
│   │   ├── Optimised_Model_Training.py
│   │   ├── Subject_Specific_Model.py
│   │   ├── bci_model.pkl
│   │   ├── evaluate_model.py
│   │   ├── model_training.py
│   │   ├── optimised_bci_model.pkl
│   │   ├── test_subjects.npy
│   │   └── testing.py
│   ├── notebooks/
│   │   ├── .ipynb_checkpoints/
│   │   ├── 01_Data_Exploration.ipynb
│   │   ├── 02_Hyperparameter_Tuning-C....ipynb
│   │   └── 03_Riemannian_Tuning.ipynb
│   ├── .DS_Store
│   ├── PROJECT_LOG.MD
│   ├── result.csv
│   └── test_subjects.npy
├── planning/
│   ├── BCI_pipeline.txt
│   └── game_frontend.txt
├── rocket_lander_game/
│   ├── __pycache__/
│   ├── assets/
│   │   └── images/
│   ├── .DS_Store
│   ├── game.py
│   ├── landing_pad.py
│   ├── main.py
│   ├── rocket.py
│   └── settings.py
├── .DS_Store
├── LICENSE
└── README.md
"""

4. Project Journey & Methodology
The project follows a systematic pipeline, from raw data ingestion to model evaluation. The methodology has evolved based on experimental results.

4.1. Dataset & Preprocessing
The project utilizes the EEG Motor Movement/Imagery Dataset from PhysioNet, containing recordings from 109 subjects. Raw EEG data is preprocessed using MNE-Python with the following steps:

Band-Pass Filtering: A filter is applied to keep frequencies between 8 Hz and 35 Hz, isolating the mu (μ) and beta (β) brain rhythms most relevant for motor imagery.

Epoching: The continuous EEG signal is segmented into discrete trials (epochs) corresponding to each motor imagery task.

4.2. Iteration 1: The CSP+SVM Approach
The first model was constructed using a standard BCI pipeline:

Feature Extraction: Common Spatial Patterns (CSP)

Classification: Support Vector Machine (SVC)

During initial implementation, a critical labeling error occurred where 'right fist' and 'both feet' were assigned the same label, inadvertently collapsing the task into a binary classification problem.

4.3. Correction and Performance Plateau
After correcting the labeling to properly represent all three classes, the CSP+SVM model was retrained and evaluated. The model achieved a classification accuracy that plateaued at approximately 45%.

While this performance is significantly better than the chance level of 33.3%, it indicates that the feature extraction capabilities of the traditional CSP algorithm have reached their ceiling with this dataset. CSP is effective at finding broad spatial patterns but may not capture the more complex temporal and spectral features necessary for higher accuracy in a 3-class scenario.

4.4. Iteration 2: Pivot to Deep Learning
Given the performance limitations of the classical machine learning approach, the project is now pivoting to a deep learning methodology. End-to-end neural network architectures, such as EEGNet or ShallowConvNet, are specifically designed for EEG data and have key advantages:

Automated Feature Extraction: These models learn the optimal features directly from the EEG data, avoiding a potential feature extraction bottleneck.

Hierarchical Feature Learning: Convolutional layers can learn a hierarchy of features, from simple frequency patterns to more complex and abstract representations that are more discriminative.

This transition is the logical next step to overcome the performance ceiling and explore the potential for a more robust and accurate classification model.

5. Future Work
This project is an ongoing effort with a clear roadmap for future development.

Implement and Train Neural Network: The immediate next step is to implement and train an EEG-specific Convolutional Neural Network (CNN) using TensorFlow/Keras to push past the 45% accuracy plateau.

Hyperparameter Tuning: Systematically tune the neural network's architecture and training parameters to maximize performance.

Real-Time BCI Implementation: Once a satisfactory accuracy is achieved, the focus will shift to a live system. This involves:

Connecting to a consumer-grade EEG headset.

Building a real-time data processing pipeline.

Integrating the trained model's output to control an external application, providing live feedback to the user.
