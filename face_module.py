
import sys
import face_recognition
import cv2, os
import numpy as np
from glob import glob
import math
from PIL import Image, ImageDraw, ImageFont


# func clear dir in images
#image_dir = 'person'


def clear_dir():
    list_images = glob('person/*')
    for list_im in list_images:
        try:
            os.remove(list_im)
        except:
            pass



def images_in_dir(input_image_dir):
    my_list = os.listdir(input_image_dir)
    images = []
    classNames = []
    for cl in my_list:
        if not cl.endswith(('mp4','.webm')):
            curl = Image.open(os.path.join(input_image_dir, cl))
            curl = np.array(curl)
            images.append(curl)
            name = cl.split('.')[0]
            classNames.append(name)
    return images, classNames

# images, classNames = images_in_dir('person')

# print(classNames)

def findEncodingFace(images):
    encodeList = list()
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# known_face_encodings = findEncodingFace(images)
# print(len(known_face_encodings))
# #
# sys.exit()

def face_as_videos(input_dir_image: str, output_path: str):
    images, classNames = images_in_dir(input_dir_image)
    known_face_encodings = findEncodingFace(images)
    file_video = glob(input_dir_image+'/*.mp4')
    if file_video:
        cap = cv2.VideoCapture(*file_video)
    else:
        print('Not File')
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    fps = 20.0
    font = ImageFont.truetype('font.otf', 15)
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    while True:
        success, img = cap.read()
        if not success: break
        pil_im = Image.fromarray(img).convert('RGB')
        draw = ImageDraw.Draw(pil_im)
        imgs = cv2.resize(img, (0,0), None, 0.5, 0.5)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgs)
        encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)
        for encodeFace, faceLocation in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(known_face_encodings, encodeFace)
            faceDistance = face_recognition.face_distance(known_face_encodings, encodeFace)
            index = np.argmin(faceDistance)
            if matches[index]:
                name = classNames[index].upper()
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                draw.text((x1-8, y1-8), name, font = font, fill=(0, 255, 255))
        out.write(np.array(pil_im))
    del draw
    out.release()