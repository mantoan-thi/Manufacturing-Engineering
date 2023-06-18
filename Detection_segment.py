from ultralytics import YOLO
import cv2
import random
import numpy as np


def overlay(image, mask, color, alpha, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    # Add object contours to the overlay
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_overlay, contours, -1, (0, 0, 0), 2)

    image_combined = cv2.addWeighted(image, 1-alpha, image_overlay, alpha, 0)

    return image_combined

def overlay2(image, mask, color, alpha, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    # Add object contours to the overlay
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_combined = cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

    #image_combined = cv2.addWeighted(image, 1-alpha, image_overlay, alpha, 0)

    return image_combined


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = (255,255,255)#color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO('Model/best_segment.pt')
class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
cap = cv2.VideoCapture('Video/Wiring Harness Testing.mp4')

output_file = 'output_video.mp4'
output_width, output_height = 1020, 780
output_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

roi_spacing = 10  # Espaçamento entre ROIs
roi_top_offset = 30  # Deslocamento vertical dos ROIs em relação ao topo

rois = []
while True:
    success, img = cap.read()
    if not success:
        break
       
    h, w, _ = img.shape
    results = model.predict(img,conf=0.5, stream=True)
    # print(results)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    rois.clear()  # Limpar a lista de ROIs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):

            seg = cv2.resize(seg, (w, h))
            #img = overlay(img, seg, colors[int(box.cls)], 1)
            img = overlay2(img, seg, (255, 0, 0), 1)

            xmin = int(box.data[0][0])
            ymin = int(box.data[0][1])
            xmax = int(box.data[0][2])
            ymax = int(box.data[0][3])

            object_label = f'{class_names[int(box.cls)]} {float(box.conf) * 100:.1f}%'
            plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], object_label)

            # Adicionar ROI (Região de Interesse) à lista de ROIs
            roi = img[ymin:ymax, xmin:xmax]
            rois.append(roi)

    #img = cv2.resize(img, (1080, 780))
    # Adicionar todos os ROIs à imagem principal
    w_offset = 0  # Deslocamento horizontal para posicionar os ROIs
    max_roi_h = 0  # Altura máxima de ROI para calcular o deslocamento vertical
    for roi in rois:
        roi_h, roi_w, _ = roi.shape
        if roi_h > max_roi_h:
            max_roi_h = roi_h

        img[roi_top_offset:roi_top_offset+roi_h, w_offset:w_offset+roi_w] = roi
        w_offset += roi_w + roi_spacing

    # Adicionar os nomes das classes abaixo dos ROIs
    text_height = 20  # Altura do texto da classe
    total_height = roi_top_offset + max_roi_h + text_height  # Altura total para ROIs e nomes das classes

    w_offset = 0  # Reiniciar o deslocamento horizontal
    contador = 1
    contador2 = 0  
    for roi, box in zip(rois, boxes):
        roi_h, roi_w, _ = roi.shape
        xmin = int(box.data[0][0])
        ymin = int(box.data[0][1])

        class_label = class_names[int(box.cls)]
        cv2.putText(img, f"Branch: {contador}", (w_offset, total_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        contador+=1
        contador2+=1
        w_offset += roi_w + roi_spacing
    cv2.rectangle(img, (2,25), (110,5), (255, 255, 255), -1, cv2.LINE_AA) 
    cv2.putText(img, f"Detections: {contador2}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow('yolov8s-seg', img)
    img = cv2.resize(img, (output_width, output_height))
    out.write(img)  # Salvar o frame no vídeo de saída
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Fechar 
#cv2.destroyAllWindows()