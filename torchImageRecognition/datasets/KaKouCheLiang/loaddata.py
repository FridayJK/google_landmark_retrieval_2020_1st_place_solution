#coding=utf-8
import math, re, os, sys, time
import PIL as Image
import xml.etree.ElementTree as ET 

Image.open()

root_path = "/root/google_landmark_retrieval_2020_1st_place_solution/torchImageRecognition/dataset/KaKouCheLiang"
xml_name = "20210902120000607_浙F9DY23.xml"
ET.parse(xml_name)
