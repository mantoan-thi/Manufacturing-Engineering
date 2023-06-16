import cv2

def save_frame(frame, count):
    # Define o nome do arquivo de imagem usando o número do frame
    filename = f"Fotos/frame_{count}.jpg"
    # Salva o frame como uma imagem JPEG
    cv2.imwrite(filename, frame)
    print(f"Salvando {filename}")

def main():
    video_path = "Dataset\Wiring Harness Testing.mp4"  # Substitua pelo caminho do seu vídeo
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    frame_count = 0
    save_count = 0

    # Obtém a taxa de quadros (FPS) do vídeo
    fps = video.get(cv2.CAP_PROP_FPS)
    # Calcula o número de frames para pular a cada 3 segundos
    frames_to_skip = int(fps * 0.5)

    while True:
        success, frame = video.read()

        if not success:
            break

        frame_count += 1
        if frame_count % frames_to_skip == 0:
            save_frame(frame, save_count)
            save_count += 1

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()