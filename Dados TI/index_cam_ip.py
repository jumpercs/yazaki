import numpy as np
import cv2
import os
from multiprocessing import Process, Queue
import time
import json
from gtts import gTTS
from playsound import playsound
import threading
import face_recognition
import sqlite3

# Tamanho do frame buffer
frame_buffer_size = 1
frame_buffer = []

# Tempo de espera para a próxima verificação
tempo_espera = 1

# Variável para controlar a verificação
ultimo_reconhecimento = time.time()

# Caminho para a pasta 'Fotos'
pasta_fotos = 'Fotos'

founded_user = ""
last_person_found = ""

# Conexão com o banco de dados
conn = sqlite3.connect('Dados_ti Banco Pronto.db')
cursor = conn.cursor()

# Arquivo JSON para armazenar as codificações
arquivo_codificacoes = 'codificacoes_faciais.json'

# Carregar as codificações faciais do banco de dados (ou do arquivo JSON se ele existir)
try:
    with open(arquivo_codificacoes, 'r') as f:
        codificacoes_faciais = json.load(f)
        known_face_encodings = codificacoes_faciais['encodings']
        known_face_names = codificacoes_faciais['names']
except FileNotFoundError:
    known_face_encodings = []
    known_face_names = []
    cursor.execute("SELECT id_rede, nome FROM dados_ti")
    for row in cursor:
        print(f"Carregando imagem para {row[0]}: {row[1]}")
        id_rede, nome = row
        try:
            imagem_banco = cv2.imread(os.path.join(pasta_fotos, f"{id_rede}.jpg"))
            codificacoes_banco = face_recognition.face_encodings(imagem_banco)[0]
            known_face_encodings.append(codificacoes_banco.tolist())  # Converta para lista para salvar no JSON
            known_face_names.append(nome)
        except Exception as e:
            print(f"Erro ao carregar imagem para {id_rede}: {nome}, Erro: {e}")

    # Salve as codificações no arquivo JSON
    codificacoes_faciais = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(arquivo_codificacoes, 'w') as f:
        json.dump(codificacoes_faciais, f)

last_frame_time = time.time()


def run():
    capture = cv2.VideoCapture("rtsp://admin:ABCtech2024@192.168.80.110:554/onvif1", cv2.CAP_FFMPEG)
    while True:
        # Ler um frame da câmera
        ret, frame = capture.read()
        if not ret:
            break
        
        
        
        # # Redimensionar o frame em 80%
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
    
        # # Adicionar o frame ao buffer
        frame_buffer.append(frame)
        if len(frame_buffer) > frame_buffer_size:
            frame_buffer.pop(0)

        # Verificar o tempo desde o último reconhecimento
        if time.time() - ultimo_reconhecimento >= tempo_espera:
            # Processar o frame mais antigo do buffer
            frame_processado = process_frame(frame_buffer[0])
        else:
            frame_processado = process_frame_only_label(frame_buffer[0])

        # Exibir o frame processado
        cv2.imshow('Câmera', frame_processado)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    global ultimo_reconhecimento
    global founded_user

    # Encontrar rostos no frame
    rostos_usuario = face_recognition.face_locations(frame)

    # Encontrar as codificações faciais
    codificacoes_usuario = face_recognition.face_encodings(frame, rostos_usuario)

    # Verificar se há rostos no frame
    if len(rostos_usuario) == 0:
        return frame

    # Loop para cada rosto detectado
    for rosto_usuario, codificacao_usuario in zip(rostos_usuario, codificacoes_usuario):
        # Comparar com as codificações faciais conhecidas
        resultados_comparacao = face_recognition.compare_faces(known_face_encodings, codificacao_usuario)
        print(resultados_comparacao)

        # Verificar se o rosto é reconhecido
        if True in resultados_comparacao:
            global last_person_found
            if time.time() - ultimo_reconhecimento >= tempo_espera:
                indice_correspondente = resultados_comparacao.index(True)
                nome_usuario = known_face_names[indice_correspondente]

                founded_user = nome_usuario
                ultimo_reconhecimento = time.time()

                tts = gTTS(text=f'Olá {founded_user}', lang="pt-br")
                filename = "voice.mp3"
                if os.path.exists(filename):
                    os.remove(filename)
                tts.save(filename)
                if last_person_found != founded_user:
                    threading.Thread(target=playsound, args=(filename,), daemon=True).start()
                    last_person_found = founded_user
        else:
            founded_user = "unknown"
            if founded_user == "unknown":
                tts = gTTS(text='Pessoa Desconhecida', lang="pt-br")
                filename = "voice.mp3"
                if os.path.exists(filename):
                    os.remove(filename)
                tts.save(filename)
                if last_person_found != founded_user:
                    threading.Thread(target=playsound, args=(filename,), daemon=True).start()
                    last_person_found = founded_user
            ultimo_reconhecimento = time.time()

    return frame

def process_frame_only_label(frame):
    cv2.putText(frame, f"Usuario: {founded_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

if __name__ == '__main__':
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    p = Process(target=run)
    p.start()
    p.join()
