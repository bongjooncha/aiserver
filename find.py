from ultralytics import YOLO

model = YOLO('./runs/detect/train2/weights/best.pt')
results=model('https://upload.wikimedia.org/wikipedia/commons/f/f9/Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg', save_txt= True, save=True)