import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

drop_value = 0.5
max_len = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
num_epochs = 30



def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            data.append(parts)

    return pd.DataFrame(data, columns=["label", "message"])

def preprocess_message(message):
    if pd.isnull(message):
        return ""
    message = str(message).lower()

    message = re.sub(r'https?://\S+|www\.\S+', '', message)

    message = re.sub(r'[^a-z0-9\s$£]', '', message)
    return message


def visualize_wordclouds(df):
    ham_msg = df[df.label == 'ham']
    spam_msg = df[df.label == 'spam']

    ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
    spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())
    # print("ham_msg_text",ham_msg_text)
    # print("spam_msg_text",spam_msg_text)
    ham_msg_cloud = WordCloud(width=320, height=160, stopwords=STOPWORDS, max_font_size=50, background_color="tomato", colormap='Blues').generate(ham_msg_text)
    plt.figure(figsize=(10, 8))
    plt.imshow(ham_msg_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    spam_msg_cloud = WordCloud(width=320, height=160, stopwords=STOPWORDS, max_font_size=50, background_color="white", colormap='autumn').generate(spam_msg_text)
    plt.figure(figsize=(10, 8))
    plt.imshow(spam_msg_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def map_word_to_embedding(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros(model.vector_size)

def preprocess_data(df, tokenizer, word2vec_model):

    df['message'] = df['message'].apply(preprocess_message)

    ham_msg = df[df.label == 'ham']
    spam_msg = df[df.label == 'spam']

    ham_msg_df = ham_msg.sample(n=len(spam_msg), random_state=0).reset_index(drop=True)
    spam_msg_df = spam_msg.reset_index(drop=True)
    # print("ham_msg shape:", ham_msg_df)
    # print("spam_msg shape:", spam_msg_df)
    msg_df = pd.concat([ham_msg_df, spam_msg_df]).reset_index(drop=True)

    train_msg = msg_df['message']
    train_label = msg_df['label'].map({'ham': 0, 'spam': 1})

    training_sequences = []
    for text in train_msg:
        words = text.lower().split()
        word_embeddings = [map_word_to_embedding(word, word2vec_model) for word in words]
        training_sequences.append(word_embeddings)
    print("word emabeding using the word2vector",word_embeddings)
    training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type, dtype='float32')
    print("traning_sequence",training_sequences )
    return training_padded, train_label

def create_model(input_dim):

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(max_len, input_dim)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_value))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_value))
    model.add(LSTM(64))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, training_padded, train_label):
    history = model.fit(training_padded, train_label, epochs=num_epochs, validation_split=0.2, verbose=2)
    return history

def evaluate_model(model, training_padded, train_label):
    predictions = (model.predict(training_padded) > 0.5).astype("int32")
    print(classification_report(train_label, predictions, target_names=['ham', 'spam']))
    cm = confusion_matrix(train_label, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def predict_message(model, tokenizer, pred_text, word2vec_model):
    pred_text = [preprocess_message(pred_text)]
    new_seq = tokenizer.texts_to_sequences(pred_text)
    padded = pad_sequences(new_seq, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    padded_embeddings = []
    for seq in padded:
        padded_embeddings.append([map_word_to_embedding(tokenizer.index_word.get(word, oov_tok), word2vec_model) for word in seq])
    padded_embeddings = np.array(padded_embeddings)

    prediction = model.predict(padded_embeddings)
    return "spam" if prediction > 0.5 else "ham"

if __name__ == "__main__":
    word2vec_model = Word2Vec.load('C:\\Users\\intel\\PycharmProjects\\pythonProject\\NLPFILES\\word2vec.model')


    data = load_data("C:\\Users\\intel\\Desktop\\Dataset\\SMSSpamCollection.txt")
    print(data.shape)
    visualize_wordclouds(data)

    tokenizer = Tokenizer(num_words=None, char_level=False, oov_token=oov_tok)
    tokenizer.fit_on_texts(data['message'])

    training_padded, train_label = preprocess_data(data, tokenizer, word2vec_model)

    model = create_model(word2vec_model.vector_size)
    history = train_model(model, training_padded, train_label)

    evaluate_model(model, training_padded, train_label)

    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "you won prize of $400000 claim it now ",
        "call me back when you are free 03128640835",
        "feel free to call and win 30000$ ",
         "my name is Ali whats your name bro ?",
         "you won 30000$  link the link below to get your gift "
    ]

    for msg in test_messages:
        prediction = predict_message(model, tokenizer, msg, word2vec_model)
        print(f"Message: {msg}\t|\tPrediction: {prediction}")
