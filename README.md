# BDP (Base Difference Procedure) Vectorization

## Description

Our project, BDP (Base Difference Procedure) Vectorization, focuses on converting variable-length sentences in English into 150-dimensional vectors. This method aims to extract deep contextual information from sentences by analyzing noun words.

We mark the first noun of the sentence and observe how its vector changes in the presence of other nouns. We then take the difference between the updated noun vector and the original first noun vector. This process generates 100-dimensional word vectors using a pretrained Wikipedia model of Gensim. Additionally, we append 50 more dimensions containing statistical context (e.g., sentence length, parts of speech proportions, special characters, etc.). The resulting vector is 150-dimensional and is used to assess the accuracy of the BDP Vectorization.

## Usage

To use our project, run the provided files in Jupyter Notebook. We have three files: average word embeddings, BDP vectors, and Doc2Vec. First, create vectors and then check accuracy using a neural network.

## Installation

Ensure you have the following dependencies installed:
- Jupyter Notebook
- Gensim
- Pandas
- NumPy
- SpaCy
- TensorFlow
- NLTK

## Environment

- Operating System: Windows 11
- RAM: 16 GB
- Architecture: 64-bit

## Contribution

Our project currently uses only parts of speech to obtain the deeper context of the sentence. To improve it, one can explore additional operations to capture differences from the original word (first noun).

## Group Information

- College Name: National Institute of Technology, Surat - 395007, Gujarat
- Course: M.Tech CSE in Data Science
- Batch: 2023-25
- Final Presentation Date: 29 April, 2024
- Subject: Natural Language Processing
- Subject Faculty: Professor Naveen Kumar
- Group Name: Sentient Scriptors
- Group Members:
  - Harsh Bari  - p23ds004
  - Aditi Das   - p23ds008
  - Aniket Pasi - p23ds020
