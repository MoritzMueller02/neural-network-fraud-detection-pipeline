# Fraud Detection Pipeline with Neural Networks

This project implements a streamlined data pipeline for tabular fraud detection using the IEEE-CIS Fraud Detection dataset from Kaggle. The primary objective is to experiment with basic neural network architectures for binary classification of fraudulent transactions.

## Overview

- **Data Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Pipeline Steps:**
  - Data ingestion and preprocessing (column selection, label encoding, feature scaling)
  - Construction of PyTorch-compatible datasets and dataloaders
  - Preparation for neural network modeling (numerical and categorical feature handling)
- **Goal:** Provide a technical foundation for rapid prototyping and experimentation with neural network models on real-world fraud data.

## Usage

The pipeline is fully contained in [`pipeline.py`](pipeline.py). It loads, preprocesses, and batches the data for model training and evaluation.

---

*This repository is intended for technical exploration and experimentation with neural network architectures on structured fraud
# neural-network-fraud-detection-pipeline