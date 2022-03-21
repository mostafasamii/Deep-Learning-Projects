import time
from helper_functions import *

# Paths
eng_image_path = 'C:\\Users\\Lenovo\\PycharmProjects\\TesseractOCR\\EnglishImages'
arab_image_path = 'C:\\Users\\Lenovo\\PycharmProjects\\TesseractOCR\\ArabicImages'


def main():
    # Calculating the total execution time
    start_time = time.time()

    # Reading Images
    images = read_images(arab_image_path)

    # detecting_characters(images)

    # Detecting Words in Images
    detecting_words(images)

    # Calculate Execution time for the program.
    print("Total Execution time = {}.".format(round(time.time()-start_time)))
    print("DONE !!")


if __name__ == '__main__':
    main()
