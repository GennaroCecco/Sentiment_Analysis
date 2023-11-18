import pickle
from keras.src.layers import BatchNormalization
from lime.lime_text import LimeTextExplainer
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from nltk import word_tokenize
from TextPreprocessor import TextPreprocessor
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
nltk.download('stopwords')

# Carica il dataset
print("Caricamento del dataset...")
dataset_all = pd.read_csv('Dataset/Twitter_Data.csv', encoding='utf-8')
nuovi_negativi_df = pd.read_csv('Dataset/OtherNegativi.csv', encoding='latin1')

dataset_all.dropna(inplace=True)

dataset_all['clean_text'] = dataset_all['clean_text'].astype(str)
dataset_all = dataset_all[dataset_all['category'] != 0]

positivi = dataset_all[dataset_all['category'] == 1]
negativi = dataset_all[dataset_all['category'] == -1]

print("#Tweet in Twitter_Data.csv: " + str(len(dataset_all)))
print("#Tweet Positivi in Twitter_Data.csv: " + str(len(positivi)))
print("#Tweet Negativi in Twitter_Data.csv: " + str(len(negativi)))
negativi = pd.concat([negativi, nuovi_negativi_df], ignore_index=True)
dataset = pd.concat([positivi, negativi], ignore_index=True)
print("#Tweet in Dataset------>: " + str(len(dataset)))
print("#Tweet Positivi in dataset: " + str(len(positivi)))
print("#Tweet Negativi in dataset: " + str(len(negativi)))

train_ratio = 0.8
df_shuffled = dataset.sample(frac=1, random_state=42)
train_idx = int(len(dataset) * train_ratio)

df_train = df_shuffled.iloc[:train_idx]
df_val = df_shuffled.iloc[train_idx:]

df = pd.concat([df_train.assign(ind="train"), df_val.assign(ind="validation")])

df['category'] = np.where(df['category'] == -1, 0, df['category'])
textprepro = TextPreprocessor()
print("Sto pulendo i dati...")
df['clean_text'] = [textprepro.clean_text(tweet) for tweet in df['clean_text']]
df['clean_text'] = [textprepro.lemmatize_text(tweet) for tweet in df['clean_text']]
print("Pulizia dati (done)")

tweet_lengths = [len(word_tokenize(tweet)) for tweet in df['clean_text']]
max_length = max(tweet_lengths)
print("Lunghezza massima tweet tokenizzati " + str(max_length))

tokenizer = Tokenizer(split=' ')
tokenizer.fit_on_texts(df['clean_text'].values)
max_features = len(tokenizer.word_index) + 1
print("Dimensione maxfeatures: " + str(max_features))
df_train, df_val = df[df["ind"].eq("train")], df[df["ind"].eq("validation")]

X_train = tokenizer.texts_to_sequences(df_train['clean_text'].values)
X_train = pad_sequences(X_train, max_length)
X_val = tokenizer.texts_to_sequences(df_val['clean_text'].values)
X_val = pad_sequences(X_val, max_length)

tokenizer_path = "tokenizer.pkl"
print("Sto salvando il tokenizer...")
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

X_val = np.lib.pad(X_val, ((0, 0), (X_train.shape[1] - X_val.shape[1], 0)), 'constant', constant_values=(0))
Y_train = np.array(pd.get_dummies((df_train['category']).values))
Y_val = np.array(pd.get_dummies((df_val['category']).values))
Y_train = Y_train.argmax(axis=1)
Y_val = Y_val.argmax(axis=1)

embed_dim = 32
lstm_out = 16
batch_size = 32
epochs = 10
dropout_rate = 0.4

print("Costruisco il modello...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=max_length))
model.add(LSTM(lstm_out, dropout=dropout_rate, recurrent_dropout=dropout_rate))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Modello costruito con successo.")
model.summary()

print("Addestro il modello con Early Stopping e ReduceLROnPlateau...")
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])
model.save('modello.h5')

if 'lr' in history.history:
    min_lr = min(history.history['lr'])
    print(f"Learning rate minimo raggiunto: {min_lr}")
else:
    print("ReduceLROnPlateau non ha avuto effetto durante l'addestramento.")

if early_stopping.stopped_epoch == 0:
    print("L'addestramento è stato completato senza early stopping.")
else:
    print(f"L'addestramento è stato interrotto all'epoca {early_stopping.stopped_epoch + 1} a causa di early stopping.")

val_accuracy_list = history.history['val_accuracy']
last_val_accuracy = val_accuracy_list[-1]
accuracy_list = history.history['accuracy']
last_accuracy = accuracy_list[-1]
percentuale_overfitting = ((last_accuracy - last_val_accuracy) / last_accuracy) * 100
numero = percentuale_overfitting
numero_formattato = "{:.2f}".format(numero)
print(f"La percentuale di overfitting è: {numero_formattato}%")

print("Evaluate su dati che il modello non ha mai visto...")
loss, accuracy = model.evaluate(X_val, Y_val)
print(f"Loss: {loss}, Accuracy: {accuracy}")

weights = model.layers[0].get_weights()[0]

# Mappa indici delle parole agli embeddings
word_index = tokenizer.word_index
word_weights = {word: weights[index][0] for word, index in word_index.items()}

# Crea un WordCloud basato sui pesi delle parole
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_weights)

# Visualizza il WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Calcola e visualizza la matrice di confusione
Y_pred = model.predict(X_val)
Y_pred = (Y_pred > 0.5)  # Converte le probabilità in previsioni binarie
confusion = confusion_matrix(Y_val, Y_pred)

# Visualizza la matrice di confusione come heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'],
            cbar=False)
plt.xlabel('Previsto')
plt.ylabel('Effettivo')
plt.title('Matrice di Confusione')
plt.show()

# Visualizza il grafico dell'accuratezza
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoca')
plt.ylabel('Accuratezza')
plt.legend(['Accuratezza Allenamento', 'Accuratezza Validazione'])
plt.show()

explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

def predict_fn(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, max_length)
    predictions = model.predict(padded_sequences)
    positive_probabilities = predictions[:, 0]
    return np.column_stack((1 - positive_probabilities, positive_probabilities))

def explore_instances_and_text_sizes(instances, text_sizes, labels):
    for instance, label in zip(instances, labels):
        for size in text_sizes:
            # Modifica le dimensioni del testo e ottieni spiegazioni
            test_instance = instance[:size]
            print("Test_instance: " + test_instance)
            test_label = label
            explanation = explainer.explain_instance(test_instance, predict_fn, num_features=20)

            print("Label reale:", "Positive" if test_label == 1 else "Negative")
            print("Previsione del modello:", "Positive" if explanation.predict_proba[1] > 0.5 else "Negative")

            print("\nSpiegazione Lime:")
            for i in range(len(explanation.as_list())):
                print(explanation.as_list()[i])

            fig = explanation.as_pyplot_figure()
            plt.show()

# Esempi di istanze in inglese e dimensioni del testo da esplorare
instances_to_explore = [
    "The product is really fantastic!",
    "I did not like it at all, terrible purchase.",
    "The service was exceptional, I highly recommend it!",
    "I'm quite satisfied with this product, it exceeded my expectations.",
    "Not worth the money, very disappointed with the quality.",
    "The delivery was prompt, but the product didn't meet my expectations."
]

labels_to_explore = [1, 0, 1, 1, 0, 0]
text_sizes_to_explore = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
explore_instances_and_text_sizes(instances_to_explore, text_sizes_to_explore, labels_to_explore)
