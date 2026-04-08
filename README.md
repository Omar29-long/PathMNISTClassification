## Academic Context

This project was conducted as part of the Data Science program at Sorbonne Université, under the supervision of Professor José Angel Garcia Sanchez.

# PathMNIST Classification

Deep learning project for multiclass classification of histopathology images from the PathMNIST dataset, with comparison of multiple architectures and detailed error analysis.

## My Contributions

- Performed exploratory data analysis (pixel statistics, class distribution, texture analysis)
- Trained and compared multiple models: MLP, CNN, ResNet18, and Vision Transformer (ViT)
- Conducted detailed error analysis with a focus on false negatives and recall for class 8 (cancerous tissue)
- Investigated inter-class similarity and model confusion patterns
- Applied interpretability techniques such as Grad-CAM
- Analyzed the impact of positional encoding in Vision Transformers

## Models Explored

- MLP (baseline, vectorized input)
- CNN (spatial feature extraction)
- ResNet18 (transfer learning)
- Vision Transformer (ViT)

## Key Results

- CNN significantly outperformed MLP due to preservation of spatial structure
- ResNet18 achieved strong performance through transfer learning
- Vision Transformer showed competitive performance despite lower parameter count
- Removing positional encoding in ViT improved performance in this specific setting
- Data augmentation did not significantly improve CNN performance

## Focus on Medical Relevance

Special attention was given to:
- False negatives (missed cancer cases)
- Recall for class 8 (critical for diagnosis)

This reflects real-world constraints where missing a positive case is more critical than a false alarm.

## Key Concepts

- Multiclass image classification
- Medical imaging (histopathology)
- Model comparison and evaluation
- Error analysis and confusion matrices
- Interpretability (Grad-CAM)

## What I learned

- Importance of spatial information in image classification tasks
- Trade-offs between model complexity and performance
- Role of interpretability in sensitive domains like healthcare
- Limitations of standard techniques (e.g., positional encoding, data augmentation) depending on data characteristics

## Original Project

This project was developed in collaboration with:
https://github.com/andrealoy/PathMNISTClassification
