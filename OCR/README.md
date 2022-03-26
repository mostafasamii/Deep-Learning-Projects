# Introduction:

OCR is the process of digitizing a document image into its constituent characters. Also it is a
complex problem because of the variety of language, fonts, and styles in which text
can be written, and the complex rules of languages. OCR generally consists of several
sub-processes to perform as accurately as possible. The sub-processes are:

  * Preprocessing of the Image
  * Text localization
  * Character segmentation
  * Character recognition
  * Post-processing

## Baseline Experiments:

In order to perform the task of OCR I followed the following pipeline:
  * Prepare Tesseract Environment
  * Reading Images: reading a single image using cv2 library to load it from the disk using cv2.imread
  * Preprocessing : Grayscale images
  * Detect the words : I started by extracting the bounding boxes coordinates of the text in the image to grab the OCR’s text itself and then draw bounding boxes around the detected text
  * Store results in flat files

## Other Experiments:
In order to enhance the preprocessing, I resized the greyscale images by half of its axes and
applied adaptive threshold which helped a lot with the detecting accuracy as it segment an image
by setting all pixels whose intensity values are above a threshold to a foreground values and all
the remaining pixels.
I also tried applying Canny Filter, Opening but it didn’t work well for detecting text in my
provided dataset, as the preprocessing differs from problem to another, they might be powerful in
another dataset.

## Tools:

  * Tesseract 0.3.9
  * Pycharm Edition 2021.3
  * Python 3.10.1
  * Anaconda Environnement

## Conclusion:

The benefit of using Tesseract to perform text detection and OCR is that we can do so in just a
single function call, making it easier than the multistage OpenCV OCR process.
In this approach I have used Tesseract which uses LSTM to do its great work. Word finding was
done by organizing text lines into blobs, and the lines and regions are analyzed for fixed pitch or
proportional text. Text lines are broken into words differently according to the kind of character
spacing. Recognition then proceeds as a two-pass process. In the first pass, an attempt is made to
recognize each word in turn. Each word that is satisfactory is passed to an adaptive classifier as
training data. The adaptive classifier then gets a change to more accurately recognize text lower
down the page.

Also the preprocessing techniques applied before the Tesseract helps in enhancing the
performance along with the transfer learning. As without the preprocessing the Tesseract will
perform quite poorly if there is a significant amount of noise or your image is not properly
preprocessed and cleaned before applying Tesseract
