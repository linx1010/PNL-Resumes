from pymongo import MongoClient
import pandas as pd
folder = 'PNL-Resumes/data/'
client = MongoClient('mongodb://localhost:27017/')
db = client['data_train']

print('Remove coleções para evitar duplicidades eventuais duplicidades')

# Verificar se a coleção existe
colecoes = db.list_collection_names()
for n in range(len(colecoes)):
    db[colecoes[n]].drop()

print('inserindo dados de treino')
df = pd.read_csv(folder+'summaries_train.csv')
collection = db['summaries_train']
json_df = df.to_dict(orient='records')
inserted_data = collection.insert_many(json_df)

print('inserindo questoes de treino')
df = pd.read_csv(folder+'prompts_train.csv')
collection = db['prompts_train']
json_df = df.to_dict(orient='records')
inserted_data = collection.insert_many(json_df)

print('inserindo questoes de teste')
df = pd.read_csv(folder+'prompts_test.csv')
collection = db['prompts_test']
json_df = df.to_dict(orient='records')
inserted_data = collection.insert_many(json_df)

print('inserindo dados de teste')
df = pd.read_csv(folder+'summaries_test.csv')
collection = db['summaries_test']
json_df = df.to_dict(orient='records')
inserted_data = collection.insert_many(json_df)



client.close()