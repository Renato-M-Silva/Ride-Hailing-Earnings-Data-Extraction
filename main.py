import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import easyocr
import time
import subprocess
import csv
import sys
import threading

"""Finds the background color of an image and returns whether it is light or dark."""
def find_background_color(image):
    # Resize the image to speed up calculation (optional)
    image = cv2.resize(image, (200, 200))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel value
    average = gray.mean()

    # Define a threshold to determine whether the color is light or dark
    threshold = 128

    if average < threshold:
        return "dark"
    else:
        return "light"

"""Opens a file selection dialog and returns a list with the paths of the selected files."""
def select_files():
    root = tk.Tk()  # Create a Tkinter window
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(
        title="Select files",  # Window title
    )

    return list(file_paths)  # Convert the returned tuple to a list

"""Adds the paths of the selected files to the list."""
def add_files_to_list(file_list):
    selected_file_paths = select_files()

    if selected_file_paths:  # Check if the user selected any files
        file_list.extend(selected_file_paths)  # Add the paths to the list
        print("Files added:")
        for file_path in selected_file_paths:
            print(file_path)
    else:
        print("No files selected.")

"""Pre-process the image and find the contours"""
def preprocess(image_to_process, background_color):
    # changes based on image been lither or darker:
    #  in grayscale we have 41 ± 5 for darker images and 244 ± 5 for lighter ones
    #  in contour detection block_size = 11 / c_constant = 11 for darker  and block_size =  / c_constant =  for ligther images
    if background_color == 'dark':
        lower_gray = 36
        upper_gray = 46
        block_size = 11
        c_constant = 11
    elif background_color =='light':
        lower_gray = 239
        upper_gray = 249
        block_size = 5
        c_constant = 5
    else:
        print('Error finding background color')

    # Grayscale
    gray_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)

    # Mask to isolate gray tones
    mask = cv2.inRange(gray_image, lower_gray, upper_gray)

    # Apply mask
    filtered_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    # Preprocessing for contour detection
    thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c_constant)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contour detection
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

"""Find horizontal lines to divide the image and returns a list"""
def find_lines(contours):
    # Filter horizontal contours
    horizontal_lines = []
    for contour in contours:
        # Approximate contour with a straight line
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        # Check if the contour is approximately horizontal
        if len(approx) == 2:
            x1, y1 = approx[0][0]
            x2, y2 = approx[1][0]
            # Adjust values according to your image
            if abs(y2 - y1) < 10 and x2 - x1 > 50:
                horizontal_lines.append(approx)
    return horizontal_lines

"""Calculate distances between lines"""
def find_distances(horizontal_lines):
    distances = []
    if len(horizontal_lines) > 1:
        for i in range(1, len(horizontal_lines)):
            previous_y = horizontal_lines[i - 1][0][0][1]
            current_y = horizontal_lines[i][0][0][1]
            distance = current_y - previous_y
            distances.append(distance)
    else:
        print("⚠ No or only one line detected. No distances to calculate.")

    # Print very long distances (arbitrary value for testing)
    for i, distance in enumerate(distances):
        if distance < -380:
            print(f"Distance: {distance}, i: {i}")

    return distances

"""Function to create subimages"""
def create_subimages(image_to_process, horizontal_lines):
    horizontal_lines.reverse()
    subimages = []
    previous_height = 0
    for i, line in enumerate(horizontal_lines):
        current_height = line[0][0][1]  # Get the y-coordinate of the line
        subimage_height = current_height - previous_height
        if subimage_height > 40:  # Check if the subimage height is greater than 40 pixels
            subimage = image_to_process[previous_height:current_height, :]  # Create the subimage
            subimages.append(subimage)
        previous_height = current_height
    return subimages

"""Reads the text in the images and returns a list of strings"""
def read_subimages(subimages):
    image_text = []
    reader = easyocr.Reader(['pt', 'en'])  # Initialize EasyOCR with the specified language
    for subimage in subimages:
        text = reader.readtext(subimage, detail=0)  # Extract text without bounding box details
        image_text.append(' '.join(text))  # Join the text parts into a single string
    return image_text

"""Remove all elements from the list that contain the words that shows that the image was not a data or an uber trip, for exemplo the word 'Histórico'"""
def remove_values(text_list):
    words_to_remove = ['Histórico', 'durante', 'Recurso']
    return [text for text in text_list if not any(word in text for word in words_to_remove)]

"""Displays a loading animation in the console until a stop event is signaled."""
def loading_animation(stop_event):
    chars = "/-\|"
    while not stop_event.is_set():
        for char in chars:
            sys.stdout.write('\r' + 'Processing GPU... ' + char)
            sys.stdout.flush()
            time.sleep(0.1)
            if stop_event.is_set():
                break

if __name__ == "__main__":
    print("Starting the process...")

    # Selecting image Files:
    my_list = []  # Create an empty list
    add_files_to_list(my_list)  # Call the function to add files and print the final list

    print("------------------------------------")

    # Starting to read the images:
    for image_file_path in my_list:
        print(f'Getting the data from image: {image_file_path}.')
        image_to_process = cv2.imread(image_file_path)

        # Find background color and Print result
        background_color = find_background_color(image_to_process)
        print(f"The image has a {background_color} background.")

        ### Calling functions ###

        contours = preprocess(image_to_process, background_color)

        """ # testing contours. if needed, Activate this code by uncommenting it.
        lower_gray = 239
        upper_gray = 249
        # Grayscale
        gray_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        # Mask to isolate gray tones
        mask = cv2.inRange(gray_image, lower_gray, upper_gray)
        # Apply mask
        filtered_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        # Preprocessing for contour detection
        for block_size in [5, 7, 9]:
            for c_constant in [2, 5]:
                thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c_constant)
                # Visualizar a imagem binária
                cv2.imshow(f"blockSize: {block_size}, c: {c_constant}", thresh[32000:32500, :])
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Contour detection
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                test_image = image_to_process.copy()
                for contour in contours:
                    cv2.drawContours(test_image, [contour], 0, (0, 255, 0), 2)
                cv2.imshow(f'test_image0-blockSize: {block_size}, c: {c_constant}', test_image[000:500, :])
                cv2.imshow(f'test_image1-blockSize: {block_size}, c: {c_constant}', test_image[32000:32500, :])
        cv2.waitKey(0)
        """
        
        print("Contours - ok!")

        horizontal_lines = find_lines(contours)
        print("Horizontal Lines - ok!")

        distances = find_distances(horizontal_lines)
        print("Distances - ok!")

        subimages = create_subimages(image_to_process, horizontal_lines)
        print("Subimages - ok!")

        start_time = time.time() # Monitor GPU usage during EasyOCR processing
        process = subprocess.Popen(['nvidia-smi', '-l', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Start nvidia-smi in a subprocess
        print("Nvidia-smi started in subprocess - ok!")
        stop_event = threading.Event()  # Create a threading.Event object to signal the animation thread to stop.
        animation_thread = threading.Thread(target=loading_animation, args=(stop_event,))  # Create a new thread to run the loading animation function, passing the stop_event as an argument.
        animation_thread.daemon = True  # Set the animation thread as a daemon thread, so it will terminate when the main thread exits.
        animation_thread.start()  # Start the animation thread.
        image_text = read_subimages(subimages)  # Process the subimages and extract text using EasyOCR.
        stop_event.set()  # Signal the animation thread to stop by setting the stop_event.
        animation_thread.join()  # Wait for the animation thread to finish its execution.
        process.terminate()  # Terminate the nvidia-smi subprocess.
        end_time = time.time()  # Record the end time of the EasyOCR processing.
        print(f"\rEasyOCR processing time: {end_time - start_time:.2f} seconds")
        print("Reading images - ok!")

        image_text = remove_values(image_text)
        print("Removing values - ok!")

        # Create CSV file with the same name and path as the image file
        base_name = os.path.splitext(image_file_path)[0]  # Remove the file extension
        csv_file_path = base_name + ".csv"

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',') #Added delimiter
            csv_writer.writerow(['Extracted Text']) #Added Header
            for text in image_text:
                encoded_text = text.encode('utf-8', errors='replace').decode('utf-8') # Encode the text to UTF-8, replacing any unencodable characters
                csv_writer.writerow([encoded_text])

        print(f"Data saved to {csv_file_path}")
        print("------------------------------------")

        """ # Testing . if needed, Activate this code by uncommenting it.
        print("------------------------------------")
        print("Testing...")
        # Print total number of lines
        print(len(horizontal_lines))

        # Show contours and lines on copies of the image
        # Draw the found contours on a copy of the image
        test_image = image_to_process.copy()
        for contour in contours:
            cv2.drawContours(test_image, [contour], 0, (0, 255, 0), 2)
        cv2.imshow('test_image', test_image[:, :])

        # Create a copy of the original image to draw the lines
        image_with_lines = image_to_process.copy()
        # Draw the detected lines and add the line number and distance to the previous line
        for i, line in enumerate(horizontal_lines):
            x1, y1 = line[0][0]
            x2, y2 = line[1][0]
            # Print line on the image
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if i > 0:
                distance = distances[i - 1]
                text = f"Line: {i}, Distance: {distance}"
                # Adjust the text position to be more centered and readable
                text_x = (x1 + 15)
                # Adjust the vertical position of the text
                text_y = y1 + 15
                # Print text of line number and distance
                cv2.putText(image_with_lines, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('image_with_lines', image_with_lines[:, :])


        # display the first 10 subimages
        for i in range(min(10, len(subimages))):  # Limit to 10 or the total number of subimages, whichever is less
            try:
                cv2.imshow(f"subimage_{i}", subimages[i])
            except cv2.error as e:
                print(f"Error displaying subimage {i}: {e}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save image with lines
        cv2.imwrite("lines.png", image_with_lines)

        # Print the first 10 lines of text
        for i in range(min(10, len(image_text))):
            print(f"Text from subimage {i + 1}: {image_text[i]}")
    """


    print("The process ended.")