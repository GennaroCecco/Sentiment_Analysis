import pickle
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
from sklearn.utils import resample

# Carica il dataset
print("Caricamento del dataset...")
dataset_all = pd.read_csv('Dataset/Twitter_Data.csv', encoding='utf-8')
dataset_all.dropna(inplace=True)
dataset_all['clean_text'] = dataset_all['clean_text'].astype(str)
dataset_all = dataset_all[dataset_all['category'] != 0]

positivi = dataset_all[dataset_all['category'] == 1]
negativi = dataset_all[dataset_all['category'] == -1]

print("#Tweet in Twitter_Data.csv: " + str(len(dataset_all)))
print("#Tweet Positivi in Twitter_Data.csv prima del sottocampionamento: " + str(len(positivi)))
print("#Tweet Negativi in Twitter_Data.csv prima del sottocampionamento: " + str(len(negativi)))

# Bilanciamo il dataset sottocampionando la rnn positiva
if len(positivi) > len(negativi):
    positivi = resample(positivi, replace=False, n_samples=len(negativi), random_state=42)
else:
    negativi = resample(negativi, replace=False, n_samples=len(positivi), random_state=42)

print("#Tweet Positivi in Twitter_Data.csv dopo il sottocampionamento: " + str(len(positivi)))
print("#Tweet Negativi in Twitter_Data.csv dopo il sottocampionamento: " + str(len(negativi)))

dataset = pd.concat([positivi, negativi], ignore_index=True)

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

tokenizer_path = 'RNN-LSTM/#1/tokenizer.pkl'
print("Sto salvando il tokenizer...")
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

X_val = np.lib.pad(X_val, ((0, 0), (X_train.shape[1] - X_val.shape[1], 0)), 'constant', constant_values=(0))
print("Dimensione di X_val dopo il padding: " + str(X_val.shape))

Y_train = np.array(pd.get_dummies((df_train['category']).values))
Y_val = np.array(pd.get_dummies((df_val['category']).values))

Y_train = Y_train.argmax(axis=1)
Y_val = Y_val.argmax(axis=1)

embed_dim = 100
lstm_out = 64
batch_size = 64
epochs = 20
dropout_rate = 0.5

print("Costruisco il modello...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-5)

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=max_length))
model.add(LSTM(lstm_out, dropout=dropout_rate, recurrent_dropout=dropout_rate))
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
