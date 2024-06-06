import face_recognition
import sqlite3
import cv2
import os
import time
import json
import threading

# Caminho para a pasta 'Fotos'
pasta_fotos = 'Fotos'

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
        id_rede, nome = row
        try:
            imagem_banco = cv2.imread(os.path.join(pasta_fotos, f"{id_rede}.jpg"))
            codificacoes_banco = face_recognition.face_encodings(imagem_banco)[0]
            known_face_encodings.append(codificacoes_banco.tolist())  # Converta para lista para salvar no JSON
            known_face_names.append(nome)
        except:
            print(f"Erro ao carregar imagem para {id_rede}: {nome}")

    # Salve as codificações no arquivo JSON
    codificacoes_faciais = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(arquivo_codificacoes, 'w') as f:
        json.dump(codificacoes_faciais, f)

# Iniciar a câmera
camera = cv2.VideoCapture(0)

# Tamanho do frame buffer
frame_buffer_size = 10
frame_buffer = []

# Tempo de espera para a próxima verificação
tempo_espera = 1  # Tempo em segundos

# Variável para controlar a verificação
ultimo_reconhecimento = 0

# Lock para sincronizar o acesso ao buffer
lock = threading.Lock()

# Variáveis para cálculo do frame rate
frame_count = 0
start_time = time.time()

# Função para processar um frame
def process_frame(frame, tempo_espera):  # Adicione tempo_espera como argumento
    print("Processando frame...")
    global ultimo_reconhecimento
    # Encontrar rostos no frame
    rostos_usuario = face_recognition.face_locations(frame)

    # Encontrar as codificações faciais
    codificacoes_usuario = face_recognition.face_encodings(frame, rostos_usuario)

    # Verificar se há rostos no frame
    if len(rostos_usuario) == 0:
        return frame

    # Verificar se a verificação pode ser realizada
    if time.time() - ultimo_reconhecimento >= tempo_espera:
        # Loop para cada rosto detectado
        for rosto_usuario, codificacao_usuario in zip(rostos_usuario, codificacoes_usuario):

            # Comparar com as codificações faciais conhecidas
            resultados_comparacao = face_recognition.compare_faces(known_face_encodings, codificacao_usuario, tolerance=0.15)

            # Verificar se o rosto é reconhecido
            if True in resultados_comparacao:
                indice_correspondente = resultados_comparacao.index(True)
                nome_usuario = known_face_names[indice_correspondente]
                print(f"Usuário encontrado: {nome_usuario}")
                cv2.putText(frame, f"Usuário: {nome_usuario}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ultimo_reconhecimento = time.time()  # Atualiza o último tempo de reconhecimento
                break  # Sai do loop após encontrar um rosto reconhecido
        else:
            print("Usuário não reconhecido.")
            cv2.putText(frame, "Usuário não reconhecido", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calcular e exibir o frame rate
    global frame_count, start_time
    frame_count += 1
    if time.time() - start_time >= 1:
        fps = frame_count / (time.time() - start_time)
        start_time = time.time()
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Loop principal
while(True):
    # Ler um frame da câmera
    ret, frame = camera.read()

    # Adicionar o frame ao buffer
    with lock:  # Adquire o lock antes de acessar o buffer
        frame_buffer.append(frame)
        if len(frame_buffer) > frame_buffer_size:
            frame_buffer.pop(0)

    # Processar o frame mais antigo do buffer em uma thread separada
    thread = threading.Thread(target=process_frame, args=(frame_buffer[0], tempo_espera))  # Passe tempo_espera
    thread.start()

    # Exibir o frame processado (pode não ser o mais recente)
    with lock:  # Adquire o lock para acessar o buffer com segurança
        if len(frame_buffer) > 0:
            frame_processado = frame_buffer[0]
            cv2.imshow('Câmera', frame_processado)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechar a conexão com o banco de dados
conn.close()

# Liberar a câmera
camera.release()

# Fechar todas as janelas
cv2.destroyAllWindows()