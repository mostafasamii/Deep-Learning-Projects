from csv import writer, DictWriter
from PIL import ImageFont, Image, ImageDraw
from preprocessing_funcs import *
import os
import cv2
from tqdm import tqdm
import pytesseract
import warnings
warnings.filterwarnings("ignore")

# Specifying the directory of installed Tesseract on MyPC
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Storing files Path
textFiles_path = 'C:\\Users\\Lenovo\\PycharmProjects\\TesseractOCR\\Results'


def read_images(path):
    '''
    INPUT: Images path
    OUTPUT: List of Images
    OBJECTIVE: Reading Images from a given path
    '''
    images = []
    for img in os.listdir(path):
        read_img = cv2.imread(os.path.join(path, img))
        resized_img = cv2.resize(read_img, None, fx=0.5, fy=0.5)
        # Preprocessing
        gray = get_grayscale(resized_img)
        adaptivethresh = adaptive_threshold(gray)
        # final_img = deskew(adaptivethresh)
        # thresh = thresholding(gray)
        # open = opening(thresh)
        # cann = canny(open)
        images.append(adaptivethresh)

    return images


def store_results_textfile(word, x, y, width, height):
    '''
    INPUT:
        word: The word was read from the given image
        x: the X-coordinate of the image
        y: the Y-coordinate of the image
        width: The width of the box
        height: The height of the box
    OUTPUT: NONE
    OBJECTIVE: Storing Image info in text file format
    '''
    image_info = [word, x, y, width, height]
    with open(textFiles_path + '\\word_coordinates.txt', 'a', encoding="utf-8") as f:
        f.write("Height     Width       Y     X       Word\n")
        for ele in image_info:
            f.write(ele)
            f.write('        ')
        f.write('\n')
    f.close()


def store_results_csv(word, x, y, width, height):
    '''
    INPUT:
        word: The word was read from the given image
        x: the X-coordinate of the image
        y: the Y-coordinate of the image
        width: The width of the box
        height: The height of the box
    OUTPUT: NONE
    OBJECTIVE: Storing Image info in CSV file format
    '''
    image_info = [word, x, y, width, height]
    # list of column names
    field_names = ['WORD', 'X', 'Y', 'WIDTH', 'HEIGHT']
    with open(textFiles_path+'\\words_coordinates.csv', 'a', newline='', encoding='utf-8') as f:
        # pass the CSV file object ot the writer() function
        writer_object = writer(f)
        # Pass the data in the list as an argument into the writerrow() function
        writer_object.writerow(image_info)
        f.close()


def detecting_characters(images):
    '''
    INPUT: List of images
    OUTPUT: Storing image data in CSV/TXT file format
    OBJECTIVE: Detecting the characters of the word
    '''
    for img in tqdm(images):
        height_img, width_img, channels = img.shape
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            b = b.split(' ')
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, height_img - y), (w, height_img - h), (0, 0, 255), 2)
            cv2.putText(img, b[0], (x, height_img - y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        # cv2.imshow('result', img)
        # cv2.waitKey(0)


def detecting_words(images):
    '''
    INPUT: List of images
    OUTPUT: Storing image data in CSV/TXT file format
    OBJECTIVE: Detecting the words in the images
    '''
    print("Working on images ...")
    for img in tqdm(images):
        height_img, width_img, = img.shape
        # Reading the image in arabic
        boxes = pytesseract.image_to_data(img, lang='ara')
        for x, b in enumerate(boxes.splitlines()):
            if x != 0:
                b = b.split()
                # print(b)
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    # Storing data in text files for each image
                    if b[0] != '-1':
                        store_results_csv(b[11], str(x), str(y), str(w), str(h))
                        store_results_textfile(b[11], str(x), str(y), str(w), str(h))
                    cv2.rectangle(img, (x, y), (width_img + x, height_img + h), (0, 0, 255), 2)
                    cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        # cv2.imshow('result', img)
        # cv2.waitKey(0)
