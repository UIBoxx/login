import cv2
import os
import glob
import pathlib
import shutil
import numpy as np
from PIL import Image
from ultralytics import YOLO

class Prediction_IMG:
    """Research and development in Object Detection YOLOv8 model training on the custom dataset"""

    def __init__(self):

        self.prediction_folder = "../models/predict"

        self.input_image_source = "../source/image"
        
        self.train_model_path =  "../models/train"

        self.active_model_path = "../models/active/best.pt"

        self.default_model_path =  "../models/default/yolov8n.pt"

        self.script_dir = pathlib.Path(__file__).parent.absolute()

        self.active_model_dir = os.path.join(self.script_dir, self.active_model_path)

        self.train_models_dir = os.path.join(self.script_dir,  self.train_model_path)

        self.default_model_dir = os.path.join(self.script_dir, self.default_model_path)

        self.input_source = os.path.join(self.script_dir, self.input_image_source)

        self.prediction_folder  = os.path.join(self.script_dir, self.prediction_folder)

    
    def get_latest_train_model(self):
        train_folders = [folder for folder in os.listdir(self.train_models_dir) if os.path.isdir(os.path.join(self.train_models_dir, folder))]
        model = "weights/yolov8n.pt"
        folder_name = train_folders[0]
        train_model_path = os.path.join(self.train_models_dir, folder_name, model)

        return  train_model_path
       
       

    def make_prediction(self, image):

        # remove old predict folder at first
        for folder_name in os.listdir(self.prediction_folder):
            folder = os.path.join(self.prediction_folder, folder_name)
            if os.path.isdir(folder):
                shutil.rmtree(folder)

       # Make prediction with latest train model
        self.latest_train_model = self.get_latest_train_model()

        if os.path.isfile(self.latest_train_model):
            # Convert the PIL Image to a NumPy array
            img_array = np.array(image)

            # Initialize the YOLO model with the latest training model
            model = YOLO(self.latest_train_model)

            # Perform predictions on the image
            results = model.predict(
                source=img_array,
                save=True,
                project=self.prediction_folder,
                name="predict",
            )


            class_list =[]
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    #r = box.xyxy[0].astype(int)
                    cls = box.cls
                    cls_name = result.names[int(cls[0])]
                    class_list.append(cls_name)

            class_dict = {}
            for item in class_list:
                class_dict[item] = class_dict.get(item, 0) + 1

            if class_dict:

                return class_dict
            else:
                no_detection = "no detections add more data and retrain"

                return  no_detection

        # if new user make folders structure for new user and save data
        else:
            image_name = os.listdir(self.input_source)
            image_url = os.path.join(self.input_source, image_name)

            model = YOLO(self.active_model_dir)
            img = Image.open(image_url)
            img_array = np.array(img)

            results = model.predict(
                source=img_array,
                save=True,
                project= self.prediction_folder,
                name="predict",
            ) 

            class_list =[]

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    #r = box.xyxy[0].astype(int)
                    cls = box.cls
                    cls_name = result.names[int(cls[0])]
                    class_list.append(cls_name)

            class_dict = {}
            for item in class_list:
                class_dict[item] = class_dict.get(item, 0) + 1
            
            if class_dict:

                return class_dict
            else:
                no_detection = "no detections add more data and retrain"
                
                return  no_detection
                         

    def get_latest_prediction_img(self):
        predict_folders = [folder for folder in os.listdir(self.prediction_folder) if os.path.isdir(os.path.join(self.prediction_folder, folder))]
        folder_name =  predict_folders[0]
        folder_path = os.path.join(self.prediction_folder, folder_name)

        image_files = os.listdir(folder_path)
        for file_name in image_files:
            image_path = os.path.join(folder_path, file_name)

        return image_path
        
prediction_img =  Prediction_IMG()

# latest_model_file = prediction.get_latest_train_model()

# make_prediction= prediction.make_prediction()
# print(make_prediction)
# prediction_file = prediction.get_latest_prediction_img()
# # print(prediction_file)


