# LSTM Language Model

This project implements a language model using an LSTM neural network to predict and generate text in English. The dataset consists of English-Spanish translations, from which English phrases are used for training and testing.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Overview

The project trains an LSTM-based language model on English text to predict the next word in a sentence. This is useful for language modeling tasks and can be extended to various applications, such as chatbots or text generation.

## Dataset

The dataset is sourced from [ManyThings.org](https://www.manythings.org/anki/spa-eng.zip), which provides English-Spanish sentence pairs. This project only uses the English phrases for language modeling. The data is preprocessed to convert Unicode characters to ASCII, normalize, and remove non-letter characters.

## Model Architecture

The model uses a bidirectional LSTM network with two layers. Key features include:

- **Input Size**: Vocabulary size of English words.
- **Hidden Size**: 512
- **Output Size**: Vocabulary size of English words.
- **Bidirectional LSTM**: To capture context in both forward and backward directions.
- **Linear Layer**: Maps the LSTM output to the vocabulary for predictions.

The model uses an embedding layer for input and an output layer to map the hidden state to word probabilities.

## Dependencies

Install dependencies by running:

    pip install torch numpy matplotlib scikit-learn

## Training

The model is trained on shuffled English sentences with the following setup:

    Loss Function: Cross-Entropy Loss
    Optimizer: Adam
    Learning Rate Scheduler: Exponential decay

Training is done over multiple epochs, with loss and accuracy tracked for train, validation, and test sets.

## Prediction

The model can predict the next word in a given sentence. Using the trained LSTM, you can generate text by feeding an initial input sequence and iteratively predicting subsequent words until the model outputs an end-of-sequence token or reaches a maximum length.
Results

Training curves are plotted to show the loss over time across the training, validation, and test sets. Example predictions include:

    Input: "what is"
    Output: "what is your name <EOS>"

    Input: "how are"
    Output: "how are you <EOS>"

## Usage

    Data Preprocessing: Download and unzip the spa-eng.zip file, then normalize the text as shown in the code.
    Training: Run train_lstm() to start training.
    Prediction: Use predict() to generate text based on an initial phrase.
