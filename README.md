# Concept Positioning System

Welcome to the Concept Positioning System project! This repository contains the code, data processing pipelines, and model implementations developed to build a personalized concept recommendation system using Stack Overflow data. This work was done as part of the Summer Research we (Aishini B, Girik Khullar, Sabari Srinivas) did at IIT Ropar from May-July 2024.

## Project Overview

This project aims to provide concept recommendations based on user knowledge. By treating each Stack Overflow tag as a "concept," we recommend learning paths tailored to individual users. The dataset, collected from Stack Overflow and available under a Creative Commons license, supports a Kaggle competition challenge.

### Core Features

- **Concept Recommender**: Recommends new tags (concepts) based on users' past interactions.
- **Dataset Creation**: Extensive preprocessing and integration of Stack Overflow data for personalized recommendations.
- **Synthetic Data Experimentation**: Explored the use of synthetic data to enhance the dataset.
- **Model Comparisons**: Implemented several recommendation models, including Alternating Least Squares (ALS), Bivariate Variational Autoencoder (BiVAE), and Collaborative Denoising Autoencoder (CDAE).

## Dataset

The dataset includes Stack Overflow interactions, with tags representing concepts users are familiar with. Data processing steps:

- **Data Extraction**: Focused on the Posts, Users, and Tags tables.
- **Filtering**: Reduced from an initial 65,000 tags to a refined set of 231, focusing on key topics.
- **Binary Matrix**: Created a user-tag matrix for model training and evaluation.
- **Challenge Dataset**: Designed to support a Kaggle competition by dividing user interactions into training and test sets.

## Synthetic Data Generation and Evaluation

We experimented with synthetic data to enrich the dataset, using **CTGAN (Conditional Tabular GAN)** to generate additional data points:

- **CTGAN Architecture**: Generated data conditioned on user features to mimic real interaction patterns.
- **Evaluation of Synthetic Data**: Compared real and synthetic data distributions, tested statistical similarity, and analyzed performance impact.
  
Ultimately, synthetic data was excluded due to limitations:

- **Data Quality**: Inability to capture fine-grained patterns observed in real interactions.
- **Bias**: Potential introduction of biases that could degrade model performance.
- **Performance Impact**: Model accuracy decreased with synthetic data augmentation, suggesting the added noise diluted the signal in the real data.

## Models

To create effective recommendations, we implemented and evaluated the following models:

- **Alternating Least Squares (ALS)**: Matrix factorization approach for collaborative filtering.
- **Bivariate Variational Autoencoder (BiVAE)**: A probabilistic model tailored for user-item interactions.
- **Collaborative Denoising Autoencoder (CDAE)**: Neural network designed for implicit feedback and top-N recommendation.

## Evaluation

All models were evaluated using the **F1 Score@5** metric, which measures precision and recall within the top-5 recommendations, aligning with the competition’s focus on identifying relevant tags.

| **Model**                                | **F1 score@5** |
|------------------------------------------|-----------------|
| Collaborative Denoising Autoencoder (CDAE) | 4%              |
| Alternating Least Squares (ALS)         | 24.96%          |
| Bivariate Variational Autoencoder (BiVAE) | 49%             |
| Bayesian Personalized Ranking (BPR)     | 50.3%           |


## Project Challenges and Learnings

This project involved handling large volumes of Stack Overflow data and experimenting with data augmentation techniques. Some key takeaways include:

- The synthetic data experiment provided insights into the challenges of data augmentation for recommendation systems.
- We learned that high data quality is crucial to avoid performance degradation from augmented datasets.
- Model comparisons highlighted the effectiveness of ALS, BiVAE, and BPR approaches for top-5 tag prediction tasks.

## License

This project uses data under a [Creative Commons License](https://creativecommons.org/licenses/).
