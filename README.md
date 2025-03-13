# TVDE_earnings-history
# Image Text Extraction Script

This Python script processes image files to extract text using OCR (Optical Character Recognition) and saves the extracted text to CSV files.

## Overview

The script is specifically designed to process screenshots from mobile phone screens displaying the earnings history of ride-hailing app drivers. 
It extracts the relevant information from these images and saves it into CSV files.

The script performs the following tasks:

1.  **Image Selection:**
    * Allows the user to select multiple image files (screenshots) using a file selection dialog.
2.  **Image Preprocessing:**
    * Determines the background color of each image to optimize preprocessing.
    * Applies grayscale conversion, masking, adaptive thresholding, and contour detection to isolate text regions.
3.  **Line Segmentation:**
    * Identifies horizontal lines in the image to segment it into subimages, corresponding to individual trip records.
    * Calculates distances between the detected lines.
4.  **Text Extraction:**
    * Uses EasyOCR to extract text from each subimage.
    * Filters out irrelevant text based on predefined keywords (e.g., "Histórico", "durante", "Recurso").
5.  **Data Storage:**
    * Saves the extracted text to CSV files, with each CSV file having the same name and path as the original image file.

## Specific Application

This script is tailored to process screenshots of ride-hailing app earnings history. These screenshots typically contain:

* Date and time of trips
* Earnings per trip
* Other relevant trip details

By processing these images, the script automates the extraction of earnings data, which can be useful for record-keeping and analysis.

## Notes

* The script uses adaptive thresholding and contour detection to identify text regions, which may require adjustments based on the characteristics of your images.

* EasyOCR supports multiple languages, and the script is configured to use Portuguese and English. You can modify the language settings in the read_subimages function.

* The script optionally monitors GPU usage during EasyOCR processing using nvidia-smi.

* The script saves the extracted text into CSV files, that have the same name and location of the images that where used.

* The script is optimized to read prints of the screen that shows the history of earnings of a driver in a app.

## Requirements

* Python 3.x
* Libraries:
    * `opencv-python` (cv2)
    * `numpy`
    * `tkinter`
    * `easyocr`
    * `nvidia-ml-py3` (optional, for GPU monitoring)

You can install the required libraries using pip:

```bash
pip install opencv-python numpy tkinter easyocr nvidia-ml-py3
