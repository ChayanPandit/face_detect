import tensorflow as tf
import zipfile, os
import splitfolders
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

    
    
def create_model():
  print("FOOO")
  splitfolders.ratio('Dataset', 
                      'Split_Data', 
                      seed=42, 
                      ratio=(0.8, 0.2))
  print("DDD")
  base_dir = 'Split_Data'
  train_dir = os.path.join(base_dir, 'train') ## Train Dataset
  validation_dir = os.path.join(base_dir, 'val') ## Validation Dataset

  # train_live_dir = os.path.join(train_dir, 'Live')
  # train_spoof_dir = os.path.join(train_dir, 'Spoof')

  # val_live_dir = os.path.join(val_dir, 'Live')
  # val_spoof_dir = os.path.join(val_dir, 'Spoof')


  train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 20,
      width_shift_range = 0.1,
      height_shift_range = 0.2,
      horizontal_flip = True,
      shear_range = 0.2,
      zoom_range = 0.2,
      fill_mode = 'nearest'
  )

  validation_datagen = ImageDataGenerator(
      rescale = 1./255,
  )

  train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size = (150, 150),
      #batch_size = 128,
      class_mode = 'binary'
  )

  validation_generator = validation_datagen.flow_from_directory(
      validation_dir,
      target_size = (150, 150),
      #batch_size = 128,
      class_mode = 'binary'
  )
      
  model_1 = tf.keras.models.Sequential([
    # first CONV => RELU => CONV => RELU => POOL layer set                                   
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(150, 150, 3)),
      #tf.keras.layers.BatchNormalization(1),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
      #tf.keras.layers.BatchNormalization(1),
      tf.keras.layers.MaxPool2D(2,2),
      tf.keras.layers.Dropout(0.25),

    # second CONV => RELU => CONV => RELU => POOL layer set
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
      #tf.keras.layers.BatchNormalization(1),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
      #tf.keras.layers.BatchNormalization(1),
      tf.keras.layers.MaxPool2D(2,2),
      tf.keras.layers.Dropout(0.25),

    # first (and only) set of FC => RELU layers
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      #tf.keras.layers.BatchNormalization(1),
      tf.keras.layers.Dropout(0.5), 

    # softmax classifier
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model_1.compile(
      loss = 'binary_crossentropy',
      optimizer = 'Adam',
      metrics = ['accuracy']
  )
  
  history_1 = model_1.fit(
      train_generator,
      epochs = 50,
      #callbacks = [reduce_LR, stop_early],
      validation_data = validation_generator,
      verbose = 1
  )
      
  def plot_accuracy(history):
    plt.figure(figsize=(18,5))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plot_acc = plt.plot(epochs, acc, 'red', label='Training Accuracy')
    plot_val_acc = plt.plot(epochs, val_acc, 'blue', label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Training and Validation Accuracy', fontsize=25)
    plt.legend(bbox_to_anchor=(1,1), loc='best')
    plt.grid()
    plt.show()

  def plot_loss(history):
    plt.figure(figsize=(18,5))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plot_loss = plt.plot(epochs, loss, 'red', label='Training Loss')
    plot_val_loss = plt.plot(epochs, val_loss, 'blue', label='Validation Loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Training and Validation Loss', fontsize=25)
    plt.legend(bbox_to_anchor=(1,1), loc='best')
    plt.grid()
    plt.show()
      
  plot_accuracy(history_1)
  plot_loss(history_1)

  model_1.save('model_1.h5')



def create_labels():
  import os
  import cv2
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from tqdm import tqdm

  # Define the class mapping (0 for "Live" and 1 for "Spoof")
  class_mapping = {
      "Uncovered": 0,
      "Hand_and_Object": 1,
      "Helmet_and_Cap": 2,
      "Mask": 3,
      "Scarf": 4,
      "Spectacles": 5,
  }

  # Function to create YOLO label file in a specified folder
  def create_label_file(image_path, class_id, bbox, label_folder):
      # Extract the filename without extension
      file_name = os.path.splitext(os.path.basename(image_path))[0]
      
      # Define the label file path in the specified folder
      label_file_path = os.path.join(label_folder, f"{file_name}.txt")
      
      with open(label_file_path, "w") as label_file:
          label_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

  # Define the paths to your image directories and label directories
  uncovered_image_dir = "Refined_Project_Dataset/Uncovered"
  uncovered_label_dir = "Refined_Project_Dataset/Uncovered"
  
  Hand_and_Object_image_dir = "Refined_Project_Dataset/Hand_and_Object"
  Hand_and_Object_label_dir = "Refined_Project_Dataset/Hand_and_Object"
  
  Helmet_and_Cap_image_dir = "Refined_Project_Dataset/Helmet_and_Cap"
  Helmet_and_Cap_label_dir = "Refined_Project_Dataset/Helmet_and_Cap"
  
  Mask_image_dir = "Refined_Project_Dataset/Mask"
  Mask_label_dir = "Refined_Project_Dataset/Mask"
  
  Scarf_image_dir = "Refined_Project_Dataset/Scarf"
  Scarf_label_dir = "Refined_Project_Dataset/Scarf"
  
  Spectacles_image_dir = "Refined_Project_Dataset/Spectacles"
  Spectacles_label_dir = "Refined_Project_Dataset/Spectacles"

  # Initialize the face detector
  detector = FaceDetector()
  
  
  dir = [
          [uncovered_image_dir,uncovered_label_dir],
          [Hand_and_Object_image_dir,Hand_and_Object_label_dir],
          [Helmet_and_Cap_image_dir,Helmet_and_Cap_label_dir],
          [Mask_image_dir,Mask_label_dir],
          [Scarf_image_dir,Scarf_label_dir],
          [Spectacles_image_dir,Spectacles_label_dir],
          ]

  # Process images in the "Live" directory
  
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

def create_yaml():
    import yaml
    import random
    import shutil
    from tqdm import tqdm
    
    dataset_dir = "Without_Hands"
    output_dir = "Without_Hands_Used"

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
    Helmet_and_Cap_images = [f for f in os.listdir(os.path.join(dataset_dir, "Helmet_and_Cap")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Mask_images = [f for f in os.listdir(os.path.join(dataset_dir, "Mask")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Scarf_images = [f for f in os.listdir(os.path.join(dataset_dir, "Scarf")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    Spectacles_images = [f for f in os.listdir(os.path.join(dataset_dir, "Spectacles")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
  
    random.shuffle(Uncovered_images)
    random.shuffle(Hand_and_Object_images)
    random.shuffle(Helmet_and_Cap_images)
    random.shuffle(Mask_images)
    random.shuffle(Scarf_images)
    random.shuffle(Spectacles_images)
    
    # Calculate total no of images
    total_images = len(Uncovered_images)+len(Hand_and_Object_images)+len(Helmet_and_Cap_images)+len(Mask_images)+len(Scarf_images)+len(Spectacles_images)
    total_train_images = total_images*train_ratio
    total_test_images = total_images*test_ratio
    total_val_images = total_images*val_ratio
    

    for [type,start,ratio] in [["train",0,train_ratio],["test",train_ratio,test_ratio],["val",train_ratio+test_ratio,val_ratio]]:
      for id,[images,folder] in enumerate([[Uncovered_images,"Uncovered"],
                                           [Hand_and_Object_images,"Hand_and_Object"],
                                           [Helmet_and_Cap_images,"Helmet_and_Cap"],
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


    dataset_dir = "Without_Hands_Used"

    # data_dict = {
    #     'path': "C:\\Users\\91858\\Desktop\\projects\\sem 8",
    #     'train': f'{os.path.join(dataset_dir, "train")}',
    #     'val': f'{os.path.join(dataset_dir, "val")}',
    #     'nc': 6,
    #     'names': ["Uncovered","Hand_and_Object","Helmet_and_Cap","Mask","Scarf","Spectacles"]
    # }

    data_dict = {
        'path': "/kaggle/input/refined-dataset/Without_Hands_Used",
        'train': "train/images",
        'val': "val/images",
        'nc': 6,
        'names': ["Uncovered","Hand_and_Object","Helmet_and_Cap","Mask","Scarf","Spectacles"]
    }
    
    with open('data.yaml', 'w') as yaml_file:
        yaml.dump(data_dict, yaml_file)

    print("Done")

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

def use_yaml():
  from ultralytics import YOLO
  
  model = YOLO('yolov8n.pt')
  
  model.train(data='data.yaml', epochs=50)


def use_yolo_model():
  
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector

  cap = cv2.VideoCapture(0)


  model = YOLO('new_masks_added.pt')
  classNames = [
      "Uncovered",
      "Hand_and_Object",
      "Helmet_and_Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  detector = FaceDetector()
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        print("Faces detected: ",len(boxes))
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
            cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                   colorB=color)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(1)

def use_yolo_with_hands():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  import shapely
  from shapely.geometry import Polygon
  import torch
  from torchvision.ops import nms



  cap = cv2.VideoCapture(0)


  model = YOLO('new_masks_added.pt')
  classNames = [
      "Uncovered",          #0
      "Hand_and_Object",    #1
      "Helmet_and_Cap",     #2
      "Mask",               #3
      "Scarf",              #4
      "Spectacles",         #5
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
  detector2 = FaceDetector()
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=False, verbose=True, iou=0.5)
    hands, img = detector.findHands(img, draw=False, flipType=True)
    
  
    for r in results:
        boxes = r.boxes
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0         
        xf, yf, wf, hf = x1,y1,0,0
        color = (0, 255, 0)
        conf1 = 0
    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"
        
        if(len(boxes)>1):
          # img, bboxs = detector2.findFaces(img,draw=True)
          print("Multiple faces")
          print("Faces detected: ",len(boxes))
          
          for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            print(x1, " ", y1," ",cls," ",conf)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
          
          break

        if(len(boxes)==0):
          print("No faces detected first time")
          img, bboxs = detector2.findFaces(img,draw=False)
          
          if(len(bboxs)>1):
            print("Multiple faces detected on second time")
            break
          if(len(bboxs)==0):
            clsname = "No face detected"
            color = (255,0,0)
            print("No face detected")
          
          for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
                      
            offsetPercentageW = 5
            offsetPercentageH = 10
            offsetW = (offsetPercentageW/100)*w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            
            offsetH = (offsetPercentageH/100)*h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 4)
            
            # Ensure that x, y, w, and h stay within image dimensions
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)
            
            x1, y1, x2, y2 = x, y, x1+w, y1+h
            
            # xf = x1
            # yf = y1
            
            # uncover = 0.7
            
            # box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            # polygon_face = Polygon(box_face)
            # total_face_area = polygon_face.area
            
            # cvzone.cornerRect(img, (x,y,w,h), (255,0,0), 3)
            color = (0,0,255)
            cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
            cvzone.putTextRect(img, "Covered",(max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,colorB=color)            
            break
        
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            
            xf,yf,wf,hf = x1,y1,w,h
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
              
            uncover = 1
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 
            # elif classNames[cls] == "Hand_and_Object":
            #   uncover = 0.85 
            
            # cvzone.putTextRect(img, f'{classNames[cls].upper()}' , (max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color, colorB=color)                            
            # clsname = classNames[cls]
  
        # Check if any hands are detected
        for hand in hands:
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
                    
            (x1, y1, w, h) = bbox1
            x2 = x1 + w
            y2 = y1 + h
                
            box_hand1 = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                
            polygon_hands = Polygon(box_hand1)
            polygon_face = polygon_face - polygon_face.intersection(polygon_hands)

            uncover = uncover * ( polygon_face.area / total_face_area )
            if uncover < 0.95:
              clsname = "Covered"
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
        covered = (clsname=="Covered")


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(2)


def use_yolo_with_hands_with_nms():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  import shapely
  from shapely.geometry import Polygon
  import torch
  from torchvision.ops import nms



  cap = cv2.VideoCapture(0)


  model = YOLO('best (3).pt')
  classNames = [
      "Uncovered",          #0
      "Hand_and_Object",    #1
      "Helmet_and_Cap",     #2
      "Mask",               #3
      "Scarf",              #4
      "Spectacles",         #5
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
  detector2 = FaceDetector()
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=False, verbose=False, iou=0.5)
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
  
    for r in results:
        boxes = r.boxes
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0         
        xf, yf, wf, hf = x1,y1,0,0
        color = (0, 255, 0)
        conf1 = 0
    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"
        
        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        print(boxes_nms.shape)
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)

        if(len(indices)>1):
          # img, bboxs = detector2.findFaces(img,draw=True)
          print("Multiple faces")
          print("Faces detected: ",len(boxes))
          
          for index in indices:
            box = boxes[index]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            print(x1, " ", y1," ",cls," ",conf)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
          
          break

        if(len(indices)==0):
          print("No faces detected first time")
          img, bboxs = detector2.findFaces(img,draw=False)
          
          if(len(bboxs)>1):
            print("Multiple faces detected on second time")
            break
          elif(len(bboxs)==0):
            clsname = "No face detected"
            color = (255,0,0)
            print("No face detected")
          else:
            for bbox in bboxs:
              x,y,w,h = bbox["bbox"]
                        
              offsetPercentageW = 5
              offsetPercentageH = 10
              offsetW = (offsetPercentageW/100)*w
              x = int(x - offsetW)
              w = int(w + offsetW * 2)
              
              offsetH = (offsetPercentageH/100)*h
              y = int(y - offsetH * 3)
              h = int(h + offsetH * 4)
              
              # Ensure that x, y, w, and h stay within image dimensions
              x = max(0, x)
              y = max(0, y)
              w = min(img.shape[1] - x, w)
              h = min(img.shape[0] - y, h)
              
              x1, y1, x2, y2 = x, y, x1+w, y1+h
              
              color = (0,0,255)
              cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
              cvzone.putTextRect(img, "Covered",(max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,colorB=color)            
              break
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            
            xf,yf,wf,hf = x1,y1,w,h
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
              
            uncover = 1
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"
              uncover = 0.6 
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 

        # Check if any hands are detected
        for hand in hands:
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
                    
            (x1, y1, w, h) = bbox1
            x2 = x1 + w
            y2 = y1 + h
                
            box_hand1 = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                
            polygon_hands = Polygon(box_hand1)
            polygon_face = polygon_face - polygon_face.intersection(polygon_hands)

            uncover = uncover * ( polygon_face.area / total_face_area )
            if uncover < 0.95:
              clsname = "Covered"
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
        covered = (clsname=="Covered")
        print(clsname)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(2)
  
def use_yolo_with_hands_with_nms2():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  import shapely
  from shapely.geometry import Polygon
  import torch
  from torchvision.ops import nms

  cap = cv2.VideoCapture(0)


  model = YOLO('new_masks_added.pt')
  classNames = [
      "Uncovered",          #0
      "Hand_and_Object",    #1
      "Helmet_and_Cap",     #2
      "Mask",               #3
      "Scarf",              #4
      "Spectacles",         #5
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
  detector2 = FaceDetector()
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=False, verbose=False, iou=0.5)
    hands, img = detector.findHands(img, draw=False, flipType=True)
    
  
    for r in results:
        boxes = r.boxes
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0         
        xf, yf, wf, hf = x1,y1,0,0
        color = (0, 255, 0)
        conf1 = 0
    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)
        total_face_area = 1

        uncover = 1
        clsname = "Covered"
        
        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        print(boxes_nms.shape)
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)

        if(len(indices)>1):
          # img, bboxs = detector2.findFaces(img,draw=True)
          print("Multiple faces")
          print("Faces detected: ",len(boxes))
          clsname = "Multiple faces"
          color = (255,0,0)

        if(len(indices)==0):
          print("No faces detected")
          clsname = "No face detected"
          color = (255,0,0)
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            
            xf,yf,wf,hf = x1,y1,w,h
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
              
            uncover = 1
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 

        # Check if any hands are detected
        for hand in hands:
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
                    
            (x1, y1, w, h) = bbox1
            x2 = x1 + w
            y2 = y1 + h
                
            box_hand1 = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                
            polygon_hands = Polygon(box_hand1)
            polygon_face = polygon_face - polygon_face.intersection(polygon_hands)

            uncover = uncover * ( polygon_face.area / total_face_area )
            if uncover < 0.95:
              clsname = "Covered"
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
        covered = (clsname=="Covered")


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(2)
  

def use_yolo_with_hands_points_only_with_nms():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  from cvzone.FaceMeshModule import FaceMeshDetector
  import shapely
  from shapely.geometry import Polygon, Point
  import torch
  from torchvision.ops import nms

  cap = cv2.VideoCapture(0)


  model = YOLO('without_hands.pt')
  classNames = [
      "Uncovered",          #0
      "Hand_and_Object",    #1
      "Helmet_and_Cap",     #2
      "Mask",               #3
      "Scarf",              #4
      "Spectacles",         #5
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=False, verbose=False, iou=0.5)
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
  
    for r in results:
        boxes = r.boxes
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0         
        xf, yf, wf, hf = x1,y1,0,0
        color = (0, 255, 0)
        conf1 = 0
    
        clsname = "Covered"
        
        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        # print(boxes_nms.shape)
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)

        if(len(indices)>1):
          # img, bboxs = detector2.findFaces(img,draw=True)
          print("Multiple faces")
          print("Faces detected: ",len(boxes))
          clsname = "Multiple faces"
          color = (255,0,0)

        if(len(indices)==0):
          print("No faces detected")
          clsname = "No face detected"
          color = (255,0,0)
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            
            xf,yf,wf,hf = x1,y1,w,h
            
              
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet_and_Cap":
              clsname = "Uncovered"
              uncover = 0.85 
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"

        # # Check if any hands are detected
        # detector3 = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)
        # img, faces = detector3.findFaceMesh(img, draw=False)
        # face_bound = []
        # boundary_points  = [
        #   10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 140, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        # ]

        # if faces:
        #   # Loop through each detected face
        #   for point in boundary_points:
        #     # cv2.circle(img, faces[0][point], 5, (255, 0, 255), cv2.FILLED)
        #     face_bound.append(faces[0][point])
            
        
        # polygon_face = Polygon(face_bound)
        # pts = np.array(face_bound,np.int32)
        # pts = pts.reshape((-1, 1, 2))
      
        # cv2.polylines(img, [pts], True, (255,0,0), 2) 
        
        detector2 = FaceDetector()
        img, bboxs = detector2.findFaces(img,draw=True)
        
        # for bbox in bboxs:
        #     x,y,w,h = bbox["bbox"]
        #     shrink = 0.9
        #     x=x+w*(1-shrink)/2
        #     y=y+h*(1-shrink)/2
        #     w=w*shrink
        #     h=h*shrink
        #     face_bound=[[x,y],[x,y+h*0.75],[x+w/3,y+h],[x+w*(2/3),y+h],[x+w,y+h*0.75],[x+w,y]]
 
        
        for bbox in bboxs:
          x,y,w,h = bbox["bbox"]
          polygon_face = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
        
        
        for hand in hands:
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # print(x1,x2,y1,y2)
            # print(lmList1)
            
            for point in lmList1:
              [x,y,w]=point
              x,y,w=int(x),int(y),int(w)
              p = Point(x,y)
              if(polygon_face.contains(p)):
                clsname="Covered"
                print("Yep")
                break
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
        covered = (clsname=="Covered")


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(2)
 
def use_yolo_with_hands_points_only_rajneesh():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  from cvzone.FaceMeshModule import FaceMeshDetector
  import shapely
  from shapely.geometry import Polygon, Point
  import torch
  from torchvision.ops import nms

  cap = cv2.VideoCapture(0)


  model = YOLO('rajneesh2.pt')
  classNames = [
      "Uncovered", 
      "Hand_and_Object",
      "Helmet",
      "Turban",
      "Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]

  
  prev_frame_time = 0
  new_frame_time = 0
  
  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=False, verbose=False, iou=0.5)
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
  
    for r in results:
        boxes = r.boxes
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0         
        xf, yf, wf, hf = x1,y1,0,0
        color = (0, 255, 0)
        conf1 = 0
    
        clsname = "Covered"
        
        boxes_nms = torch.tensor([[int(box.xyxy[0][i]) for i in range(4)] for box in boxes])
        # print(boxes_nms.shape)
        boxes_nms = boxes_nms.reshape(-1, 4)
        confs_nms = torch.tensor([math.ceil((box.conf[0] * 100)) / 100 for box in boxes])
        boxes_nms = boxes_nms.float()
        confs_nms = confs_nms.float()

        indices = nms(boxes = boxes_nms, scores = confs_nms, iou_threshold=0.2)

        if(len(indices)>1):
          # img, bboxs = detector2.findFaces(img,draw=True)
          print("Multiple faces")
          print("Faces detected: ",len(boxes))
          clsname = "Multiple faces"
          color = (255,0,0)

        if(len(indices)==0):
          print("No faces detected")
          clsname = "No face detected"
          color = (255,0,0)
        
        for index in indices:
            box = boxes[index]
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            conf1 = conf
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            
            xf,yf,wf,hf = x1,y1,w,h
            
              
            clsname = "Covered"
            
            if classNames[cls] == "Uncovered":
              clsname = "Uncovered"
            elif classNames[cls] == "Spectacles":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Helmet":
              uncover = 0.85 
            elif classNames[cls] == "Turban":
              clsname = "Uncovered"
            elif classNames[cls] == "Cap":
              clsname = "Uncovered"
            elif classNames[cls] == "Scarf":
              clsname = "Uncovered"
              uncover = 0.9 
            elif classNames[cls] == "Mask":
              uncover = 0.6 
            elif classNames[cls] == "Hand_and_Object":
              clsname = "Uncovered"

        # # Check if any hands are detected
        # detector3 = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)
        # img, faces = detector3.findFaceMesh(img, draw=False)
        # face_bound = []
        # boundary_points  = [
        #   10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 140, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        # ]

        # if faces:
        #   # Loop through each detected face
        #   for point in boundary_points:
        #     # cv2.circle(img, faces[0][point], 5, (255, 0, 255), cv2.FILLED)
        #     face_bound.append(faces[0][point])
            
        
        # polygon_face = Polygon(face_bound)
        # pts = np.array(face_bound,np.int32)
        # pts = pts.reshape((-1, 1, 2))
      
        # cv2.polylines(img, [pts], True, (255,0,0), 2) 
        
        detector2 = FaceDetector()
        img, bboxs = detector2.findFaces(img,draw=True)
        
        # for bbox in bboxs:
        #     x,y,w,h = bbox["bbox"]
        #     shrink = 0.9
        #     x=x+w*(1-shrink)/2
        #     y=y+h*(1-shrink)/2
        #     w=w*shrink
        #     h=h*shrink
        #     face_bound=[[x,y],[x,y+h*0.75],[x+w/3,y+h],[x+w*(2/3),y+h],[x+w,y+h*0.75],[x+w,y]]
 
        
        for bbox in bboxs:
          x,y,w,h = bbox["bbox"]
          polygon_face = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
        
        
        for hand in hands:
            # Information for the first hand detected
            hand1 = hand  # Get the first hand detected
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)                
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # print(x1,x2,y1,y2)
            # print(lmList1)
            
            for point in lmList1:
              [x,y,w]=point
              x,y,w=int(x),int(y),int(w)
              p = Point(x,y)
              if(polygon_face.contains(p)):
                clsname="Covered"
                print("Yep")
                break
            
        if clsname == "Covered":
          color=(0,0,255)
        
        cvzone.cornerRect(img, (xf, yf, wf, hf),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{clsname}',(max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,colorB=color)
        
        covered = (clsname=="Covered")


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(2)
 


def use_yolo_without_hands():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector
  from cvzone.HandTrackingModule import HandDetector
  
  import shapely
  from shapely.geometry import Polygon



  cap = cv2.VideoCapture(0)


  model = YOLO('new_masks_added.pt')
  classNames = [
      "Uncovered",
      "Hand_and_Object",
      "Helmet_and_Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()

    results = model(img, stream=True, verbose=False)    
    
    for r in results:
        boxes = r.boxes
        print("Faces detected: ",len(boxes))
        # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
        
        x1, y1, x2, y2 = 0,0,0,0
         
        xf = x1
        yf = y1
        color = (0, 255, 0)
    
        box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        polygon_face = Polygon(box_face)

        uncover = 1
        
        if(len(boxes)>1):
          print("Multiple faces")
          continue

        if(len(boxes)==0):
          print("No faces detected")
          detector2 = FaceDetector()
          img, bboxs = detector2.findFaces(img,draw=True)
          
          for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
                      
            offsetPercentageW = 10
            offsetPercentageH = 20
            offsetW = (offsetPercentageW/100)*w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            
            offsetH = (offsetPercentageH/100)*h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 4)
            
            # Ensure that x, y, w, and h stay within image dimensions
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)
            
            
            x1 = x
            y1 = y
            x2 = x1 + w
            y2 = y1 + h
            
            xf = x1
            yf = y1
            
            uncover = 0.7
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            
            cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)


        clsname = ""
        
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            color = (0, 255, 0)
            
            # print(box.cls)
            
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            cvzone.cornerRect(img, (x1, y1, w, h),colorC=color,colorR=color)
            
            
            xf = x1
            yf = y1
            
            box_face = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
            polygon_face = Polygon(box_face)
            total_face_area = polygon_face.area
            
            uncover = 1
            if classNames[cls] == "Spectacles":
              uncover = 0.9 * conf
            elif classNames[cls] == "Helmet_and_Cap":
              uncover = 0.85 * conf
            elif classNames[cls] == "Scarf":
              uncover = 0.9 * conf
            elif classNames[cls] == "Mask":
              uncover = 0.6 * conf
            elif classNames[cls] == "Hand_and_Object":
              uncover = 0.85 * conf
              
            # cvzone.putTextRect(img, f'{classNames[cls].upper()}' ,
                              #  (max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,
                              #  colorB=color)
            clsname = classNames[cls]

            
            
        cvzone.putTextRect(img, f'{clsname} , {int(uncover*100)}%',
                               (max(0, xf), max(35, yf)), scale=2, thickness=4,colorR=color,
                               colorB=color)
        cover = 1- uncover


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(1)

def use_yolo_after_zoom():
  
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector

  
  cap = cv2.VideoCapture(0)
  detector = FaceDetector()

  
  model = YOLO('new_masks_added.pt')
  classNames = [
      "Uncovered",
      "Hand_and_Object",
      "Helmet_and_Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]
  
  while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img,draw=False)
    
    if bboxs:
      # bboxInfo - "id","bbox","score","center"
      
      offsetPercentageW = 10
      offsetPercentageH = 20
      
      for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        # print(x,y,w,h)
        
        offsetW = (offsetPercentageW/100)*w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        
        offsetH = (offsetPercentageH/100)*h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 4)
        
        # Ensure that x, y, w, and h stay within image dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)
        
        # cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
        
        cropped_face = img[y:y+h, x:x+w]
        
        results = model(cropped_face, stream=False, verbose=True)
        
        color = (255, 0, 0)
        conf = 0
        cls = -1
        
        for r in results:          
          boxes = r.boxes
          print("Faces detected: ",len(boxes))
          # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
          for box in boxes:
            # cvzone.putTextRect(img, f'{len(boxes)} Faces', scale=2, thickness=4, colorR=(255,0,0), colorB=(255,0,0))
                    
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
                            
            # print(box.cls)
            print(f"Class = {classNames[cls]}, Confidence = {conf}")
            cvzone.cornerRect(img, (x, y, w, h),colorC=color,colorR=color)
            cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                    (max(0, x), max(35, y)), scale=2, thickness=4,colorR=color,
                                    colorB=color)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

def test_yolo_model():
  from ultralytics import YOLO
  import time
  import tqdm

  import os
  import math
  
  test_dir = "Modified_Without_Hands_Used/test/images"
  
  model = YOLO('rajneesh2.pt')
  classNames = [
      "Uncovered", 
      "Hand_and_Object",
      "Helmet",
      "Turban",
      "Cap",
      "Mask",
      "Scarf",
      "Spectacles",
  ]

  
  print("Model Loaded")
  
  
  correct = 0
  test_files = os.listdir(test_dir)
  total_test = len(test_files)
  
  pbar_live = tqdm.tqdm(total=total_test)
  
  for root, dirs, files in os.walk(test_dir):
      for i,file in enumerate(files):
          if file.lower().endswith((".jpg", ".jpeg", ".png")):
              cls1 = -1
              if file.lower().endswith((".jpg", ".png")):
                cls1 = int(file[len(file)-5])
              else:
                cls1 = int(file[len(file)-6])
              
              # print(cls1)
                
              image_path = os.path.join(root, file)
              img = cv2.imread(image_path)
  
              results = model(img, stream=True, verbose=False)
              for r in results:
                  boxes = r.boxes
                  
                  for box in boxes:
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    # print(cls,"!!",cls1)
                    
                    if ( cls==cls1):
                      correct += 1

              pbar_live.set_postfix_str(f"Correctness: {correct / (i+1) * 100:.2f}%")
              pbar_live.update(1)

  pbar_live.close()
            
  print("Test accuracy: ",(correct/total_test)*100, "%")
  
  
          
def use_keras_model():
  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector

  cap = cv2.VideoCapture(0)


  # Load your .h5 model using TensorFlow
  model = tf.keras.models.load_model('keras_model_200.h5',compile=False)
  classNames = ["Live","Spoof"]
  
  prev_frame_time = 0
  new_frame_time = 0
  
  detector = FaceDetector()
  
  while True:
    new_frame_time = time.time()
    success, img = cap.read()
    
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    
    # Predicts the model
    prediction = model.predict(image)
    cls = np.argmax(prediction)
    class_name = classNames[cls]
    confidence_score = prediction[0][cls]

    img, bboxs = detector.findFaces(img)
    
    if bboxs:
      # bboxInfo - "id","bbox","score","center"
      
      offsetPercentageW = 10
      offsetPercentageH = 20
      
      for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        # print(x,y,w,h)
        
        offsetW = (offsetPercentageW/100)*w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        
        offsetH = (offsetPercentageH/100)*h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 3.5)
        
        cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)

        color = (0, 255, 0)
          
        # print(box.cls)
        if confidence_score > 0.6:

          if classNames[cls] == 'Live':
              color = (0, 255, 0)
              print(f"Live Confidence = {confidence_score}, Class = {cls}")
          else:
              color = (0, 0, 255)
              print(f"Spoof Confidence = {confidence_score}, Class = {cls}")

        cvzone.cornerRect(img, (x, y, w, h),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(confidence_score*100)}%',
                                   (max(0, x), max(35, y)), scale=2, thickness=4,colorR=color,
                                   colorB=color)
    
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS = {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(1)

def use_keras_after_zoom():

  from ultralytics import YOLO
  import time
  import math
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector

  
  cap = cv2.VideoCapture(0)
  detector = FaceDetector()

  
  # Load your .h5 model using TensorFlow
  model = tf.keras.models.load_model('keras_model_200.h5',compile=False)
  classNames = ["Live","Spoof"]
  
  
  while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img,draw=False)
    
    if bboxs:
      # bboxInfo - "id","bbox","score","center"
      
      offsetPercentageW = 10
      offsetPercentageH = 20
      
      for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        # print(x,y,w,h)
        
        offsetW = (offsetPercentageW/100)*w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        
        offsetH = (offsetPercentageH/100)*h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 4)
        
        # Ensure that x, y, w, and h stay within image dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)
        
        # cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
        
        cropped_face = img[y:y+h, x:x+w]
        
            
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        
        # Predicts the model
        prediction = model.predict(image)
        cls = np.argmax(prediction)
        class_name = classNames[cls]
        confidence_score = prediction[0][cls]
        
        color = (0, 255, 0)
          
        # print(box.cls)
        if confidence_score > 0.6:

          if classNames[cls] == 'Live':
              color = (0, 255, 0)
              print(f"Live Confidence = {confidence_score}, Class = {cls}")
          else:
              color = (0, 0, 255)
              print(f"Spoof Confidence = {confidence_score}, Class = {cls}")

        cvzone.cornerRect(img, (x, y, w, h),colorC=color,colorR=color)
        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(confidence_score*100)}%',(max(0, x), max(35, y)), scale=2, thickness=4,colorR=color,colorB=color)
        
    cv2.imshow("Image", img)
    cv2.waitKey(25)



def use_model():
  # Load the saved model
  model_path = 'model_1.h5'
  loaded_model = load_model(model_path)

  # Define the val data directory
  val_dir = 'Dataset'  

  # Create a data generator for val data
  val_datagen = ImageDataGenerator(rescale=1./255)

  val_generator = val_datagen.flow_from_directory(
      val_dir,
      target_size=(150, 150),
      # batch_size=32,  # You can adjust the batch size as needed
      class_mode='binary'
  )

  # Evaluate the model on the val data
  evaluation = loaded_model.evaluate(val_generator)

  print("Validation Loss:", evaluation[0])
  print("Validation Accuracy:", evaluation[1])
  
  # # Get the list of filenames in the same order as the generator
  # filenames = val_generator.filenames

  # # Evaluate the model and show individual predictions
  # for i in range(len(filenames)):
  #   img, label = val_generator[i]
  #   prediction = loaded_model.predict(img)
  #   class_label = "Live" if prediction[0][0] > 0.5 else "Spoof"
  #   print(f"Image: {filenames[i]}, True Label: {label[0]}, Predicted Label: {class_label}, Prediction Score: {prediction[0][0]}")

def predict_single_image():
  # Load the pre-trained model
  loaded_model = tf.keras.models.load_model('model_1.h5')

  # Load and preprocess the image
  img = tf.keras.preprocessing.image.load_img('Dataset/Live/105.jpg', target_size=(150, 150))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  img_array /= 255.0  # Normalize the image data

  # Make the prediction
  prediction = loaded_model.predict(img_array)
  
  print(prediction)

  # Interpret the prediction
  if prediction[0][0] > 0.5:
    class_label = "Live"
    confidence = prediction[0][0]
  else:
    class_label = "Spoof"
    confidence = 1.0 - prediction[0][0]

  print(class_label, confidence)


def open_camera():
  
  import cvzone
  from cvzone.FaceDetectionModule import FaceDetector

  
  cap = cv2.VideoCapture(0)
  detector = FaceDetector()
  
  while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    
    if bboxs:
      # bboxInfo - "id","bbox","score","center"
      
      offsetPercentageW = 10
      offsetPercentageH = 20
      
      for bbox in bboxs:
        x,y,w,h = bbox["bbox"]
        # print(x,y,w,h)
        
        offsetW = (offsetPercentageW/100)*w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        
        offsetH = (offsetPercentageH/100)*h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 3.5)
        
        cv2.rectangle(img, (x,y,w,h), (255,0,0), 3)
      
    cv2.imshow("Image", img)
    cv2.waitKey(1)
  
def hands():
  import cvzone
  from cvzone.HandTrackingModule import HandDetector
  import cv2

  # Initialize the webcam to capture video
  # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
  cap = cv2.VideoCapture(0)

  # Initialize the HandDetector class with the given parameters
  detector = HandDetector(staticMode=False, maxHands=10, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

  # Continuously get frames from the webcam
  while True:
      # Capture each frame from the webcam
      # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
      success, img = cap.read()

      # Find hands in the current frame
      # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
      # The 'flipType' parameter flips the image, making it easier for some detections
      hands, img = detector.findHands(img, draw=True, flipType=True)

      # Check if any hands are detected
      if hands:
          # Information for the first hand detected
          hand1 = hands[0]  # Get the first hand detected
          lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
          bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
          center1 = hand1['center']  # Center coordinates of the first hand
          handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

          # Count the number of fingers up for the first hand
          fingers1 = detector.fingersUp(hand1)
          print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

          # Calculate distance between specific landmarks on the first hand and draw it on the image
          length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                    scale=10)

          # Check if a second hand is detected
          if len(hands) == 2:
              # Information for the second hand
              hand2 = hands[1]
              lmList2 = hand2["lmList"]
              bbox2 = hand2["bbox"]
              center2 = hand2['center']
              handType2 = hand2["type"]

              # Count the number of fingers up for the second hand
              fingers2 = detector.fingersUp(hand2)
              print(f'H2 = {fingers2.count(1)}', end=" ")

              # Calculate distance between the index fingers of both hands and draw it on the image
              length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                        scale=10)

          print(" ")  # New line for better readability of the printed output

      # Display the image in a window
      cv2.imshow("Image", img)

      # Keep the window open and update it for each frame; wait for 1 millisecond between frames
      cv2.waitKey(1)  



# hands()

# create_model()

# create_labels()
# create_labels2()
# create_yaml()
# create_yaml2()
# use_yaml()
# use_yolo_model()
# use_yolo_without_hands()
# use_yolo_with_hands()
# use_yolo_with_hands_with_nms()
# use_yolo_with_hands_with_nms2()
# use_yolo_with_hands_points_only_with_nms()
# use_yolo_with_hands_points_only_rajneesh()

# use_yolo_after_zoom()

# use_keras_model()
# use_keras_after_zoom()
test_yolo_model()
# create_and_use_yaml()
# use_model()
# predict_single_image()
# open_camera()