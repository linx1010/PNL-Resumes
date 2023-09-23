from flask import Flask, request, Response
from flask_restx import Api, Resource, fields

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from datetime import datetime

import threading

from transformers import BertForSequenceClassification, BertTokenizer
import torch


app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='API para avaliação de resumos em inglês',
    description='Uma API simples para avaliação de resumos em inglês',
)

client = MongoClient('mongodb://localhost:27017/')
#tentar criar modelos que utilizem cloudevents
# Modelo de resumo
summary_model = api.model('automatic_evaluation', {
    'id': fields.Integer,
    'student_id': fields.Integer,
    'prompt_id': fields.Integer,
    'text': fields.String,
    "content": fields.Float(default=0.0),
    "wording": fields.Float(default=0.0)
})

#Modelo para GET com o parâmetro 'id'
summary_get = api.model('automatic_evaluation_get', {
    'id': fields.Integer(description='ID do aluno')
})
summaries = []

consumo_namespace = api.namespace('consumption', description='Ambiente de consumo e resposta da avaliação')


@consumo_namespace.route('/automatic_evaluation')
class SummaryList(Resource):
    @api.doc(params={'id': 'ID do aluno'})  # Adicione esta linha
    # Criar endpoint de retorno de notas baseado no id informado
    @api.marshal_list_with(summary_model)
    def get(self):
        """Lista todos os resumos"""
        student_id = request.args.get('id')

        # Certifique-se de que o student_id seja um número inteiro
        try:
            student_id = int(student_id)
        except (ValueError, TypeError):
            # Trate o caso em que o valor não pode ser convertido em inteiro (por exemplo, não é um número)
            return 'O parâmetro "id" deve ser um número inteiro.', 400  # Retorna código de status 400 (Bad Request)

        summaries = list_summaries(student_id)
     
        return summaries

    @api.expect(summary_model)
    @api.marshal_with(summary_model, code=201)
    def post(self):
        """Cria um novo resumo"""
        new_summary = api.payload
        result = persist_resume(new_summary)
        return result

    

#Grava o resumo enviado pelo aluno
def persist_resume(summary):
    db = client['data_resumes']
    collection = db['summaries_send']
    json_df = summary
    json_df['timestamp'] = datetime.now()
    try:
        inserted_data = collection.insert_one(json_df)
    except PyMongoError as e:
        print(f"Ocorreu um erro durante a inserção: {e}")

    # robot_review(json_df['timestamp'],inserted_data.inserted_id)
    if inserted_data.acknowledged:
        # Inicie a função robot_review em uma thread separada
        review_thread = threading.Thread(target=robot_review, args=(json_df['text'], inserted_data.inserted_id))
        review_thread.start()
        return json_df,201
    else:
        return json_df,500
    


def list_summaries(student_id):
    print(student_id)
    db = client['data_resumes']
    collection = db['summaries_send']
    filter={'student_id': int(student_id)}
    result = collection.find(filter=filter)

    summaries_list = []

    # Itere pelos documentos e adicione-os à lista
    for doc in result:
        summaries_list.append({
            'id': doc['id'],
            'student_id': doc['student_id'],
            'prompt_id': doc['prompt_id'],
            'text': doc['text'],
            'content': doc['content'],
            'wording': doc['wording'],
            'timestamp': doc['timestamp']
        })
    # caso não encontre nada retorna 204
    if not summaries_list:
        return summaries_list, 204

    return summaries_list, 200


def robot_review(submitted_text,inserted_id):
    db = client['data_resumes']
    collection = db['summaries_send']

    filter = {'_id': inserted_id}
    
    
    # Caminho onde o modelo foi salvo
    folder = 'data\modelos'

    # Carregar modelo e tokenizador
    model = BertForSequenceClassification.from_pretrained(folder)
    tokenizer = BertTokenizer.from_pretrained(folder)

    # Tokenizar e codificar o texto
    input_text = tokenizer(submitted_text, padding=True, truncation=True, return_tensors='pt', max_length=128)

    # Fazer a avaliação
    model.eval()
    with torch.no_grad():
        outputs = model(**input_text)
        notas_content, notas_wording = outputs.logits[0]

    update = {'$set': {'content': notas_content.item(),
                       'wording': notas_wording.item()}}
    

    # Use update_one() para atualizar o documento
    resultado = collection.update_one(filter, update)

    # Verifique se a atualização foi bem-sucedida
    if resultado.modified_count > 0:
        print("Registro atualizado com sucesso.")
    else:
        print("Nenhum registro foi atualizado.")

    print(f"Nota de Conteúdo: {notas_content.item()}")
    print(f"Nota de Linguagem: {notas_wording.item()}")






if __name__ == '__main__':
    app.run()