# Spam Detection Web Application

## Overview
This project is a web application for detecting spam messages using machine learning techniques. It leverages natural language processing (NLP) and emoji analysis to classify messages as spam or not spam. The application is built using Flask for the backend and includes a machine learning model trained using scikit-learn.

## Features
- **Message Preprocessing**: Tokenization, lemmatization, and stopword removal
- **Emoji Analysis**: Detects and counts spam-related emojis
- **TF-IDF Vectorization**: Transforms text data for training
- **Machine Learning Model**: Uses a Random Forest classifier
- **Flask Web Interface**: Provides an easy-to-use interface for training and testing messages

## Installation
### Prerequisites
Ensure you have Python installed (recommended version: 3.7+).

### Install Required Libraries
Run the following command to install dependencies:
```sh
pip install pandas scikit-learn nltk flask numpy emoji mysql-connector-python
```

### Run the Project
Run the following command to run the project:

```sh
cd <project derectory>
```
```sh
flask run
```