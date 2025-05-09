# # -*- coding: utf-8 -*-
# """ortho.ipynb
#
# Automatically generated by Colab.
#
# Original file is located at
#     https://colab.research.google.com/drive/1BxrEW9gNvP31gwCtZZxpahAKdhKIlKqm
# """
#
# !pip install ultralytics
# !pip install yolov11
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# !ls /content/drive/MyDrive
#
# from ultralytics import YOLO
#
# model = YOLO('/content/yolo11n.pt')
#
# # Use the model
# results = model.train(data='/content/drive/MyDrive/BoneFractureYolo8/data.yaml', epochs=3)
#
# from google.colab import files
# import shutil
# shutil.make_archive('/content/runs/detect/train4', 'zip', '/content/runs/detect/train4')
# files.download('/content/runs/detect/train4.zip')
#
