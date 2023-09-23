from pymongo import MongoClient
import pandas as pd
import os
from datetime import datetime



client = MongoClient('mongodb://localhost:27017/')
db = client['data_train']
collection = db['modelos_treinados']

# # tentativa de gravar no mongo o modelo, porém o limite de arquivo é de 16mb
# model_dir = 'data/modelos'
# model_files = {}

# for root, dirs, files in os.walk(model_dir):
#     for nome_arquivo in files:
#         caminho_arquivo = os.path.join(root,nome_arquivo)
#         with open(caminho_arquivo,'rb') as arquivo:
#             model_files[nome_arquivo] = arquivo.read()

# data_hora = datetime.now()

# modelo_documento={
#     'model_name':'BERT v1.0',
#     'files':model_files,
#     'timestamp':data_hora
# }
# collection.insert_one(modelo_documento)

# client.close()
