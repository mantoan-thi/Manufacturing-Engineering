from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO('Model/best.pt')
cap = cv2.VideoCapture('Video/Wiring Harness Testing.mp4')

object_count = {}  # Dicionário para rastrear a contagem de objetos por classe

# Configuração do vídeo de saída
output_file = 'output_video.mp4'
output_width, output_height = 1020, 780
output_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

while True:
    _, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img, conf=0.65)

    object_count.clear()  # Limpar o dicionário de contagem de objetos a cada quadro

    for r in results:
        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # obter coordenadas da caixa no formato (superior, esquerda, inferior, direita)
            c = box.cls
            class_name = model.names[int(c)]
            annotator.box_label(b, class_name, color=(255, 11, 0), txt_color=(255, 255, 0))  # Modificar o tamanho da fonte para 0.4

            # Rastrear a contagem de objetos por classe
            if class_name in object_count:
                object_count[class_name] += 1
            else:
                object_count[class_name] = 1

            # Desenhar uma linha entre o objeto e o texto da contagem
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            text_x, text_y = 10, 30  # Coordenadas do texto de contagem
            line_start = ((x1 + x2) // 2, (y1 + y2) // 2)  # Ponto inicial da linha (centro do objeto)
            line_end = (text_x, text_y+120)  # Ponto final da linha (coordenadas do texto de contagem)
            cv2.line(frame, line_start, line_end, (255, 11, 0), 2)  # Desenhar a linha

    frame = annotator.result()
    frame = cv2.resize(frame, (output_width, output_height))

    # Desenhar contagem de objetos na tela
    font = cv2.FONT_HERSHEY_SIMPLEX
    for class_name, count in object_count.items():
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (text_x, text_y), font, 0.7, (255, 11, 0), 1, cv2.LINE_AA)
        text_y += 30

    cv2.imshow('YOLO V8 Detection', frame)
    out.write(frame)  # Salvar o frame no vídeo de saída

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Fechar o
