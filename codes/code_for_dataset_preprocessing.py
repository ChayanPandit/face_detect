import tensorflow as tf
import zipfile, os
import splitfolders
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

    
def create_labels2():
  import os
  import cv2
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from tqdm import tqdm

  classNames = [
      "Uncovered",        # 0
      "Hand_and_Object",  # 1
      "Helmet",           # 2
      "Turban",           # 3
      "Cap",              # 4
      "Mask",             # 5
      "Scarf",            # 6
      "Spectacles",       # 7
  ]

  # Function to create YOLO label file in a specified folder
  def create_label_file(image_path, class_id, bbox, label_folder):
      # Extract the filename without extension
      file_name = os.path.splitext(os.path.basename(image_path))[0]
      
      # Define the label file path in the specified folder
      label_file_path = os.path.join(label_folder, f"{file_name}.txt")
      
      with open(label_file_path, "w") as label_file:
          label_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

  # Define the paths to your image directories and label directories
  uncovered_image_dir = "Modified_Without_Hands/Uncovered"
  uncovered_label_dir = "Modified_Without_Hands/Uncovered"
  
  Hand_and_Object_image_dir = "Modified_Without_Hands/Hand_and_Object"
  Hand_and_Object_label_dir = "Modified_Without_Hands/Hand_and_Object"
  
  Helmet_image_dir = "Modified_Without_Hands/Helmet"
  Helmet_label_dir = "Modified_Without_Hands/Helmet"
  
  Turban_image_dir = "Modified_Without_Hands/Turban"
  Turban_label_dir = "Modified_Without_Hands/Turban"
  
  Cap_image_dir = "Modified_Without_Hands/Cap"
  Cap_label_dir = "Modified_Without_Hands/Cap"
  
  Mask_image_dir = "Modified_Without_Hands/Mask"
  Mask_label_dir = "Modified_Without_Hands/Mask"
  
  Scarf_image_dir = "Modified_Without_Hands/Scarf"
  Scarf_label_dir = "Modified_Without_Hands/Scarf"
  
  Spectacles_image_dir = "Modified_Without_Hands/Spectacles"
  Spectacles_label_dir = "Modified_Without_Hands/Spectacles"

  # Initialize the face detector
  detector = FaceDetector()
  
  
  dir = [
          [uncovered_image_dir,uncovered_label_dir],
          [Hand_and_Object_image_dir,Hand_and_Object_label_dir],
          [Helmet_image_dir,Helmet_label_dir],
          [Turban_image_dir,Turban_label_dir],
          [Cap_image_dir,Cap_label_dir],
          [Mask_image_dir,Mask_label_dir],
          [Scarf_image_dir,Scarf_label_dir],
          [Spectacles_image_dir,Spectacles_label_dir],
          ]
  
  # Remove all labels
  for id,[image_dir,label_dir] in enumerate(dir):
    for root, dirs, files in os.walk(label_dir):
        for file in tqdm(files):
            if file.lower().endswith((".txt")):
              os.remove(os.path.join(root, file))
              
  print("Removed labels")
                
  # Rename all iamges to form numer 0-n
  for id,[image_dir,label_dir] in enumerate(dir):
    for root, dirs, files in os.walk(image_dir):
      for i,f in enumerate(files):
          f_ext = os.path.splitext(f)[1]
          s = classNames[id] + "_" + str(i) + f_ext
          absname = os.path.join(root, f)
          newname = os.path.join(root, s)
          if os.path.isfile(newname)==False:
            os.rename(absname, newname)

  print("Renamed")
    
  # Create labels
  for id,[image_dir,label_dir] in enumerate(dir):
    for root, dirs, files in os.walk(image_dir):
        for file in tqdm(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)

                # Detect faces in the image
                img, bboxs = detector.findFaces(img)

                # Process the detected faces
                for bbox in bboxs:
                    x, y, w, h = bbox["bbox"]  # Get the bounding box coordinates
                    score = bbox["score"][0]
                    if score > 0.4:
                      # ------  Adding an offset to the face Detected --------
                      
                      offsetPercentageW = 10
                      offsetPercentageH = 20
                      offsetW = (offsetPercentageW / 100) * w
                      x = int(x - offsetW)
                      w = int(w + offsetW * 2)
                      offsetH = (offsetPercentageH / 100) * h
                      y = int(y - offsetH * 3)
                      h = int(h + offsetH * 3.5)

                      # ------  To avoid values below 0 --------
                      if x < 0: x = 0
                      if y < 0: y = 0
                      if w < 0: w = 0
                      if h < 0: h = 0
                      
                      # ------  Normalize Values  --------
                      ih, iw, _ = img.shape
                      xc, yc = x + w / 2, y + h / 2

                      xcn, ycn = round(xc / iw, 6), round(yc / ih, 6)
                      wn, hn = round(w / iw, 6), round(h / ih, 6)
                      # print(xcn, ycn, wn, hn)
                      
                      # ------  To avoid values above 1 --------
                      if xcn > 1: xcn = 1
                      if ycn > 1: ycn = 1
                      if wn > 1: wn = 1
                      if hn > 1: hn = 1

                    # Determine the class
                    class_id = id

                    # Create YOLO label file in the "Live/labels" folder
                    create_label_file(image_path, class_id, [xcn, ycn, wn, hn], label_dir)

  print("Label files created successfully.")

def create_yaml2():
    import yaml
    import random
    import shutil
    from tqdm import tqdm
    
    dataset_dir = "Modified_Without_Hands"
    output_dir = "Modified_Without_Hands_Used"

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2  # You can adjust this ratio

    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "labels"), exist_ok=True)

    # Lists to hold the shuffled image file names for Live and Spoof
    Uncovered_images = [f for f in os.listdir(os.path.join(dataset_dir, "Uncovered")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Hand_and_Object_images = [f for f in os.listdir(os.path.join(dataset_dir, "Hand_and_Object")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Helmet_images = [f for f in os.listdir(os.path.join(dataset_dir, "Helmet")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Turban_images = [f for f in os.listdir(os.path.join(dataset_dir, "Turban")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Cap_images = [f for f in os.listdir(os.path.join(dataset_dir, "Cap")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Mask_images = [f for f in os.listdir(os.path.join(dataset_dir, "Mask")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Scarf_images = [f for f in os.listdir(os.path.join(dataset_dir, "Scarf")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Spectacles_images = [f for f in os.listdir(os.path.join(dataset_dir, "Spectacles")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
  
    random.shuffle(Uncovered_images)
    random.shuffle(Hand_and_Object_images)
    random.shuffle(Helmet_images)
    random.shuffle(Turban_images)
    random.shuffle(Cap_images)
    random.shuffle(Mask_images)
    random.shuffle(Scarf_images)
    random.shuffle(Spectacles_images)
    
    # Calculate total no of images
    total_images = len(Uncovered_images)+len(Hand_and_Object_images)+len(Helmet_images)+len(Mask_images)+len(Turban_images)+len(Cap_images)+len(Scarf_images)+len(Spectacles_images)
    total_train_images = total_images*train_ratio
    total_test_images = total_images*test_ratio
    total_val_images = total_images*val_ratio
    

    for [type,start,ratio] in [["train",0,train_ratio],["test",train_ratio,test_ratio],["val",train_ratio+test_ratio,val_ratio]]:
      for id,[images,folder] in enumerate([[Uncovered_images,"Uncovered"],
                                           [Hand_and_Object_images,"Hand_and_Object"],
                                           [Helmet_images,"Helmet"],
                                           [Turban_images,"Turban"],
                                           [Cap_images,"Cap"],
                                           [Mask_images,"Mask"],
                                           [Scarf_images,"Scarf"],
                                           [Spectacles_images,"Spectacles"]]):
          l=len(images)
          images1 = images[int(start*l):int((start+ratio)*l)]
          for image in tqdm(images1):
              src = os.path.join(dataset_dir, folder, image)
              image_name, image_ext = os.path.splitext(image)
              image1 = f"{image_name}_{id}{image_ext}"
              dest = os.path.join(output_dir, type, "images", image1)
              shutil.copy(src, dest)

              # Get the label file path based on the image file name
              label = f"{image_name}.txt"
              label1 = f"{os.path.splitext(image1)[0]}.txt"
              src = os.path.join(dataset_dir, folder, label)
              dest = os.path.join(output_dir, type, "labels", label1)
              if os.path.exists(src):
                  shutil.copy(src, dest)


    dataset_dir = "Modified_Without_Hands_Used"

    # data_dict = {
    #     'path': "C:\\Users\\91858\\Desktop\\projects\\sem 8",
    #     'train': f'{os.path.join(dataset_dir, "train")}',
    #     'val': f'{os.path.join(dataset_dir, "val")}',
    #     'nc': 6,
    #     'names': ["Uncovered","Hand_and_Object","Helmet_and_Cap","Mask","Scarf","Spectacles"]
    # }

    data_dict = {
        'path': "/kaggle/input/modified-dataset/Modified_Without_Hands_Used",
        'train': "train/images",
        'val': "val/images",
        'nc': 8,
        'names': ["Uncovered","Hand_and_Object","Helmet","Turban","Cap","Mask","Scarf","Spectacles"]
    }
    
    with open('data.yaml', 'w') as yaml_file:
        yaml.dump(data_dict, yaml_file)

    print("Done")


create_labels2()
create_yaml2()