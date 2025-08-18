Project Log: Transition from CSP+SVM to Neural Networks
1. Initial Approach and Correction of a Classification Error
The initial objective of this project was to develop a 3-class Brain-Computer Interface (BCI) to classify three distinct motor imagery tasks: left fist, right fist, and both feet.

The first model was constructed using a standard BCI pipeline consisting of Common Spatial Patterns (CSP) for feature extraction and a Support Vector Machine (SVC) for classification. However, during the initial implementation, a critical labeling error occurred. The 'right fist' and 'both feet' classes were both assigned the numerical label 2, while 'left fist' was assigned label 1.

This inadvertently collapsed the problem into a binary classification task, where the model was only learning to distinguish 'left fist' from the combined 'right fist/feet' class. This fundamental error was identified and corrected by re-assigning the 'both feet' task to a unique label, 3, thereby restoring the intended 3-class structure of the problem.

2. Performance Plateau of the Corrected CSP+SVM Model
After correcting the labeling scheme, the CSP+SVM model was retrained and evaluated on the 3-class problem. The model achieved a classification accuracy that plateaued at approximately 45%.

While this performance is significantly better than the chance level of 33.3%, it suggests that the feature extraction capabilities of the traditional CSP algorithm have reached their ceiling with this dataset. CSP is effective at finding broad spatial patterns that are maximally discriminant between classes, but it may not be capturing the more complex, nuanced temporal and spectral features inherent in the EEG signals that are necessary for higher accuracy.

3. Justification for Pivot to Neural Networks
Given the performance limitations of the classical machine learning approach, the decision has been made to pivot to a deep learning methodology. End-to-end neural network architectures, such as EEGNet or ShallowConvNet, are specifically designed for EEG data and have several key advantages:

Automated Feature Extraction: Unlike the CSP+SVM pipeline, these models learn the optimal features directly from the raw or minimally-processed EEG data, avoiding a potential feature extraction bottleneck.

Hierarchical Feature Learning: Convolutional layers can learn a hierarchy of features, from simple frequency and spatial patterns to more complex and abstract representations that are more discriminative.

This transition is the logical next step to overcome the performance ceiling encountered and to explore the potential for a more robust and accurate classification model.