from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.models import Sequential
import seaborn as sns
import pandas as pd
import numpy as np
import os


def func(row):
    """
    distinguish novel name with number

    :param row: novel name
    :return: label
    """
    if(row["label"] == "Alice in Wonderland"):
        return 0
    elif(row["label"] == "Botchan"):
        return 1
    elif (row["label"] == "Bushido"):
        return 2
    elif (row["label"] == "The Life and Adventures of Robinson Cruesoe"):
        return 3
    elif (row["label"] == "Les Miserables"):
        return 4
    elif (row["label"] == "Les Trois Mousquetaires"):
        return 5


def plot_confusion_matrix(cm):
    """
    plot confusion matrix

    :param cm: confusion matrix
    """
    plt.figure(figsize=(15, 15))
    fig, ax = plt.subplots()
    labels = ['Alice', 'Botchan', 'Bushido', 'Les Miserables', 'Les Trois Mousquetaires',
              'Robinson Cruesoe']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.plot()
    plt.savefig("cm.png")

def plot_history(history):
    """
    summarize history for accuracy
    
    :param history: learning history created by model.fit
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.rcParams["font.size"] = 18
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()


# make the list of "label" and "text"
pd.set_option("display.max_colwidth", 100)
os.chdir("./textdata")
dataset_df = pd.read_csv("textdataset.txt", sep='\t', header=None)
dataset_df.rename({0: 'label', 1: 'text'}, axis=1, inplace=True)
dataset_df['category'] = dataset_df.apply(func, axis=1)
dataset_df.head()

# divide label and text into train and text
X_train, X_test, Y_train, Y_test = train_test_split(
    dataset_df[['text']], dataset_df[['category']],
    test_size=0.2, random_state=0
)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Create a dictionary by separating one text into word strings and create a vector converted to the index of each word
max_len = 100  # max number og 1 sentence (pudding a shortage)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['text'])
x_train = tokenizer.texts_to_sequences(X_train['text'])
x_test = tokenizer.texts_to_sequences(X_test['text'])
for text, vector in zip(X_train['text'].head(3), x_train[0:3]):
    print(text)
    print(vector)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
print(x_train[0])


# Hyperparameters
hyperparameters_scope = {'learning_rate': 0.01, 'maxEpoch': 30, 'batch_size': 32}

# create the model
vocabulary_size = len(tokenizer.word_index) + 1  # vocabulary of dataset + 1
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=32))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# set test data
y_train = Y_train['category'].values
y_test = Y_test['category'].values
one_hot_y_train = np.eye(6)[y_train]
one_hot_y_test = np.eye(6)[y_test]

# learning
history = model.fit(
    x_train, one_hot_y_train, batch_size=hyperparameters_scope['batch_size'], epochs=hyperparameters_scope['maxEpoch'],
    validation_data=(x_test, one_hot_y_test)
)

# validate text data
y_pred = model.predict_classes(x_test)
one_hot_y_pred = np.eye(6)[y_pred]
labels = ['Alice', 'Botchan', 'Bushido', 'Les Miserables', 'Les Trois Mousquetaires', 'The Life and Adventures of Robinson Cruesoe']
cm = confusion_matrix(y_test, y_pred)
cm_p = np.zeros((6,6))
# calculate accuracy
for i in range(6):
    for j in range(6):
        cm_p[i][j] = cm[i][j] / np.sum(cm, axis=1)[i]


# Plot confusion matrix
plot_confusion_matrix(cm_p)

# Plot accuracy
plot_history(history)

