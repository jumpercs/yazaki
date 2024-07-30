import face_recognition
import sqlite3
from flask import Flask, request, jsonify
import time
import json
import os
import base64
import io
from PIL import Image

app = Flask(__name__)
# mudar porta para 9999
app.config['PORT'] = 9999

# Conexão com o banco de dados SQLite
conn = sqlite3.connect('Dados_ti Banco Pronto.db')  # Substitua pelo nome correto do seu banco
cursor = conn.cursor()

# Carregar imagens de referência e codificações faciais do banco de dados
known_face_encodings = []
known_face_names = []

cursor.execute("SELECT id_rede, nome FROM dados_ti")
for row in cursor.fetchall():
    id_rede = row[0]
    nome = row[1]

    # Carregar imagem do usuário (assumir que as fotos estão em 'Dados TI/Fotos/')
    try:
        image = face_recognition.load_image_file(f"Fotos/{id_rede}.jpg")
        encoding = face_recognition.face_encodings(image)[0]  # Obter a primeira face detectada
        known_face_encodings.append(encoding)
        known_face_names.append(nome)
    except FileNotFoundError:
        print(f"Imagem não encontrada para o ID: {id_rede}")

# Fechar a conexão com o banco
conn.close()

@app.route('/reconhecer', methods=['POST'])
def reconhecer_rosto():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Nenhuma imagem em base64 enviada'}), 400

    image_base64 = data['image']

    try:
        print(image_base64)
        #save base64 to disk
        os.makedirs('uploads', exist_ok=True)
        # random temporary file name
        file_name =    'txt' + str(int(time.time())) + '.txt'
        with open(os.path.join('uploads', file_name), 'w') as f:
            f.write(image_base64)

        # Decodificar a imagem base64
        image_data = base64.b64decode(image_base64)
        # Converter a imagem para o formato necessário pelo face_recognition
        # Salva a imagem no disco
        os.makedirs('uploads', exist_ok=True)
        # random temporary file name
        file_name =    'image' + str(int(time.time())) + '.jpg'
        with open(os.path.join('uploads', file_name), 'wb') as f:
            f.write(image_data)


        image = face_recognition.load_image_file('uploads/{}'.format(file_name))
        print('uploads/{}'.format(file_name))

        #DEBUG - load image from uploads/upload_image.jpg
        # image = face_recognition.load_image_file('Fotos/20112611.jpg')


        # Obter as codificações faciais da imagem
        face_encodings = face_recognition.face_encodings(image)


        if len(face_encodings) == 0:
            print("Nenhuma face encontrada na imagem")
            return jsonify({'error': 'Nenhuma face encontrada na imagem'}), 400

        # Comparar com as faces conhecidas
        match_found = False
        nome_correspondente = None
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            
            if True in matches:
                match_found = True
                # Obter o nome da primeira correspondência
                first_match_index = matches.index(True)
                nome_correspondente = known_face_names[first_match_index]
                break  # Parar na primeira correspondência

        if match_found:
            print(f"Correspondência encontrada: {nome_correspondente}")
            return jsonify({'match': True, 'nome': nome_correspondente})
        else:
            print("Nenhuma correspondência encontrada")
            return jsonify({'match': False})

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=app.config['PORT']) 
