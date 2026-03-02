########################utils.py ######################################################################
#                                                                                                     #
# A collection of utility functions for image dataset management, visualization, and model evaluation.#
# Author: Rose Aupepin , Andréa Loy , Alizée Robin, Omar Zeroual                                      #  
#                                                                                                     #
#######################################################################################################

# Standard library imports
import os
import random
import shutil
import hashlib
import imghdr
import torch


# Data manipulation and numerical arrays
import numpy as np
import pandas as pd
import seaborn as sns

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
from PIL import Image

# Machine Learning and Deep Learning
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score
)

from typing import Union

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def display_medmnist_samples(dataset, class_filter=None, n_samples=5):
    X = dataset.imgs
    Y = dataset.labels.flatten()
    label_dict = dataset.info['label']
    
    # 1. Gestion du filtrage
    if class_filter is None or class_filter == '*':
        target_classes = np.unique(Y)
    else:
        if isinstance(class_filter, str):
            match = [int(k) for k, v in label_dict.items() if v.lower() == class_filter.lower()]
            if not match:
                print(f"Classe '{class_filter}' non trouvée.")
                return
            target_classes = match
        else:
            target_classes = [class_filter]

    n_rows = len(target_classes)
    # On ajuste la taille pour que les titres individuels respirent
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 3, n_rows * 3.5))
    
    axes = np.atleast_2d(axes)

    # 3. Remplissage
    for i, class_id in enumerate(target_classes):
        class_name = label_dict[str(class_id)]
        indices = np.where(Y == class_id)[0]
        
        n_to_show = min(n_samples, len(indices))
        # Sélection aléatoire
        if n_to_show > 0:
            selected_idx = np.random.choice(indices, n_to_show, replace=False)
        else:
            selected_idx = []

        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(selected_idx):
                idx = selected_idx[j]
                
                # Affichage de l'image
                ax.imshow(X[idx], cmap='gray' if X[idx].ndim == 2 else None)
                
                # --- AFFICHAGE DU TITRE PAR IMAGE ---
                # On met le nom de la classe en gras et l'index en dessous
                ax.set_title(f"{class_name.upper()}\nIdx: {idx}", fontsize=9, fontweight='bold')
            
            # On cache les axes mais on garde les titres
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def print_dataset_structure(dataset):
    print(f"--- Structure du Dataset : {dataset.flag.upper()} ---")
    print(f"Type d'objet : {type(dataset)}")
    print(f"Nombre total d'échantillons : {len(dataset)}")
    
    # Dimensions des images (N, H, W) ou (N, H, W, C)
    print(f"Format des images (shape) : {dataset.imgs.shape}")
    print(f"Type de données (dtype) : {dataset.imgs.dtype}")
    
    # Dimensions des labels
    print(f"Format des labels : {dataset.labels.shape}")
    
    # Vérifier si c'est du Gris ou de la Couleur
    channels = "Gris" if dataset.imgs.ndim == 3 else "RGB (Couleur)"
    print(f"Mode visuel : {channels}")

def show_class_distribution(dataset):
    labels = dataset.labels.flatten()
    label_dict = dataset.info['label']
    
    # Compter les occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Créer un tableau propre avec les noms de classes
    dist = []
    for u, c in zip(unique, counts):
        dist.append({
            "ID": u,
            "Classe": label_dict[str(u)],
            "Nombre": c,
            "Pourcentage": f"{(c/len(labels)*100):.2f}%"
        })
    
    df = pd.DataFrame(dist)
    print("\n--- Répartition des classes ---")
    print(df.to_string(index=False))
