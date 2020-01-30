import os

SQUAD_BASE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

project_dir = r'C:/Users/sbhatnagar4/Desktop/machine comprehension/Match-LSTM-Keras'
data_dir = os.path.join(project_dir, 'data')

train_filename = 'train-v2.0.json'
dev_filename = 'dev-v2.0.json'

glove_base_url = "http://nlp.stanford.edu/data/"
glove_filename = "glove.6B.zip"