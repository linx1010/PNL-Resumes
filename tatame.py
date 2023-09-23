import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch.optim as optim
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

print('inicio das atividades '+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
folder = 'data\modelos\bert'
client = MongoClient('mongodb://localhost:27017/')
db = client['data_train']
collection = db['prompts_train']
cursor = collection.find()
df_questions = pd.DataFrame(list(cursor))

collection = db['summaries_train']
cursor = collection.find()
df_sumaries = pd.DataFrame(list(cursor))

merged_df = df_questions.merge(df_sumaries, on='prompt_id')

print(merged_df.shape)

print('Extraindo os dados relevantes')
text_data = merged_df['text'].tolist()  
print('preparado os Textos das redações')
content_data = merged_df['content'].tolist()  
print('preparado Notas (conteúdo) das redações')
wording_data = merged_df['wording'].tolist()  
print('preparada as Notas (linguagem) das redações')
prompt_question_data = merged_df['prompt_question'].tolist()  
print('preparada as Perguntas associadas')
prompt_text_data = merged_df['prompt_text'].tolist()  
('preparado os Textos de prompt associados')

print('Carregar modelo BERT e tokenizador para o inglês')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 rótulos: content e wording
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print('Prepararando as entradas para o modelo')
input_texts = [f"{text} {prompt_question} {prompt_text}" for text, prompt_question, prompt_text in zip(text_data, prompt_question_data, prompt_text_data)]

print('Tokenizando e codificar os textos')
encoded_texts = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)

print('Convertendo notas de content e wording para tensores do PyTorch')
notas_content = torch.tensor(content_data, dtype=torch.float32)
notas_wording = torch.tensor(wording_data, dtype=torch.float32)

print('Concatenando as notas para criar um tensor de rótulos (target)')
rotulos = torch.stack((notas_content, notas_wording), dim=1)

print('Definindo otimizador e treinamento')
# optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)


batch_size = 8  # Tamanho do lote
gradient_accumulation_steps = 2  # Acumulação de gradientes a cada 2 mini-lotes
print('inicio do treinamento '+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    for batch_start in range(0, len(encoded_texts['input_ids']), batch_size):
        batch_end = batch_start + batch_size
        batch_inputs = {k: v[batch_start:batch_end] for k, v in encoded_texts.items()}
        batch_labels = rotulos[batch_start:batch_end]
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        if (batch_start // batch_size) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    print(f"Época {epoch+1}, Perda: {loss.item()} "+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print('fim do treinamento '+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f'Salvando o modelo em {folder}')
model.save_pretrained(folder)
tokenizer.save_pretrained(folder)