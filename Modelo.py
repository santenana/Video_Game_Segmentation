def Model_segmentation(epochs,imgsz,batch):    
    import detectron2
    from detectron2.utils.logger import setup_logger
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import random
    import roboflow 
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.data.catalog import DatasetCatalog
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg.yaml')  # Se arma un modelo a partir de YAML
    model = YOLO('yolov8n-seg.pt')  # Se usa un modelo pre entrenado para la segmentacion/deteccion de imagnes
    model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # crear un modelo haciendo transfer learning a partir del modelo .pt pre entrenado

    # Entrenar modelo
    results = model.train(data='/home/santenana/EspecializacionIA/Bimestre03/Computer Vision/lab3/dataset/data.yaml', epochs=epochs, imgsz=imgsz,batch=batch)
    #Se entrena el modelo asignandole la data de data.yml, se entrena durante 50 epocas, que es lo recomendado para 
    #modelos grandes, se ajusta los parametros de entrenamiento al hacer uso de la funcion
    
    best_model = YOLO('/home/santenana/EspecializacionIA/Bimestre03/Computer Vision/lab3/runs/segment/train10/weights/best.pt')
    # Se hace uso del mojor modelo entrenado el cual se guarda autimaticamante en la ruta
    
    return best_model # Se regresa el resultado del mejor modelo