from ultralytics import YOLO

#YOLO11l, taken from Ultralytics, trained on PascalVOC data which should be available from Ultralytics as well

def main():
    model = YOLO("yolo11l.pt")


    results = model.train(
        data="VOC.yaml",
        epochs=50,
        imgsz=512, 
        batch=-1,         
        amp=True,
        save=True,
    )


    trained_model = YOLO("runs/detect/train/weights/best.pt")

    predictions = trained_model.predict(source="EntireSet", conf=0.25, iou=0.5,save= True, save_txt=True,save_conf=True)

if __name__ == '__main__':
    main()