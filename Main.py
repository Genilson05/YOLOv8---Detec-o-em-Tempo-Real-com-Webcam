import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

print("Rodando... aperte Q na janela para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nao consegui ler a webcam")
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()
    cv2.imshow("YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
