import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dados de exemplo
texts = ["Este é um exemplo de texto completo.", "Outro texto de exemplo."]
summaries = ["Exemplo de resumo coerente com o texto completo.", "Este resumo não está coerente."]

# Rótulos de avaliação de conteúdo (valores contínuos)
content_labels = [0.8, 0.4]

# Rótulos de avaliação de gramática (valores contínuos)
wording_labels = [0.9, 0.2]

# Tokenização dos textos e resumos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts + summaries)

text_sequences = tokenizer.texts_to_sequences(texts)
summary_sequences = tokenizer.texts_to_sequences(summaries)

# Padronização das sequências para que tenham o mesmo comprimento
max_sequence_length = max(len(seq) for seq in text_sequences + summary_sequences)
text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post')
summary_sequences = pad_sequences(summary_sequences, maxlen=max_sequence_length, padding='post')

# Construção do modelo de avaliação de conteúdo
model_content = Sequential()
model_content.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_sequence_length))
model_content.add(LSTM(100))
model_content.add(Dense(1, activation='linear'))  # Saída é um valor contínuo

model_content.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Usando MSE como função de perda

# Treinamento do modelo de avaliação de conteúdo
content_labels = np.array(content_labels)
model_content.fit(text_sequences, content_labels, epochs=10, batch_size=2)

# Construção do modelo de avaliação de gramática
model_wording = Sequential()
model_wording.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_sequence_length))
model_wording.add(LSTM(100))
model_wording.add(Dense(1, activation='linear'))  # Saída é um valor contínuo

model_wording.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Usando MSE como função de perda

# Treinamento do modelo de avaliação de gramática
wording_labels = np.array(wording_labels)
model_wording.fit(summary_sequences, wording_labels, epochs=10, batch_size=2)

# Certifique-se de que a pasta de destino existe
model_dir = "notebooks/data/modelos/tensor"
os.makedirs(model_dir, exist_ok=True)

# Salvar o modelo de avaliação de conteúdo na pasta
model_content.save(os.path.join(model_dir, "model_content.h5"))

# Salvar o modelo de avaliação de gramática na pasta
model_wording.save(os.path.join(model_dir, "model_wording.h5"))

print("Modelos treinados e salvos com sucesso na pasta data/modelos/tensor.")
