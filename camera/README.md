# Lab 2: Image Data Collection with ESP32-S3 Sense

The importance of a good dataset for machine learning can't be overstated, and the quality of your model is directly dependent on the quality of the data it was trained on. In order to achieve a reliable, representative, balanced and documented dataset, careful planning before data collection is essential.

In this lab, you will write a data collection plan with labeling guidelines for a computer vision application of your choice. If you have decided on an application to build for the course project, you're welcome to use that application here and make your data collection plan part of your project documentation. After you have made a plan, you will use the provided ESP32 and Python applications to capture some images according to the plan. *Note that a two-hour session is not enough for writing a complete plan and building a complete dataset, so aim for spending one hour on each part, only to get started.*

## Prerequisites

You will need:
 * Working Python and ESP32 toolchains on your computer. The setup is described in detail in the Keyword Spotter lab.
 * An assembled ESP32-S3 Sense board with the camera extension board.
 * A USB-C cable to connect the board to your computer.
 * An idea for a simple computer vision application.

## Part 1: Write a data collection plan

Your document should at least contain the following sections:

1. *Application*. Describe the application and its purpose. Specify its functionality, under what conditions it should work, and how it should respond to valid input.
2. *Classes and labeling guidelines*. This section addresses the *reliability* quality attribute of your dataset. Specify clearly the classes that your model will distinguish between. Use your imagination and identify edge cases. For example, for a face classifier, specify how the classifier should handle a face that's partially out of view, or if there are multiple faces in the image. Write the specification such that it can be used to label any image unambiguously.
3. *Conditional factors*. This section addresses the *representative* quality attribute of your dataset. List all the conditional factors that are allowed to vary within the application scope, and for each factor, what the variations can be. For example, for a face classifier, some conditional factors include lighting, angle, indoors vs outdoors, clothing and hairstyle. Also remember to specify variations for the negative class(es) - for a face classifier, this could be background scenes, animals, or variations in faces that the classifier should reject.
4. *Collection strategy*. This section addresses both the *representative* and *balanced* qualities of your dataset. Describe how you will collect data to ensure representation and balance across classes and conditional factors. As a minimum, outline image counts or proportions over classes and factors.
5. *Documentation*. This section addresses the *documented* quality of your dataset. Specify the metadata attributes that you will tag each image or collection session with, and where and in what format you will document them.
6. *Sources*. Describe your image acquisition methods and specify the sources. If you capture images with cameras, specify which cameras you'll use, and if you use public datasets, specify which ones.
7. *Privacy and legal considerations*. Describe how you will ensure that you comply with privacy and legal requirements. This is especially important if you include photos of other people in your dataset.

When you're done with your data collection plan and labeling guidelines, you're welcome to hand them in for review, but it's not mandatory.

## Part 2: Capture images with ESP32-S3 Sense

This lab includes a very simple application for capturing images with the ESP32-S3 Sense board and transferring and saving them on your computer. It consists of an ESP32 application in the `esp32` folder and a Python application in the `python` folder, meant to be run together:
 * `esp32`: This application initializes the camera, waits for an incoming 'S' character on the serial port, and then takes one 320x200 RGB565 photo every second and sends the raw image bytes over the serial port, preceded by the preamble sequence `"\n===FRAME===\n"`.
 * `python`: This application opens a window, sends 'S' on the serial port, and then receives and displays the incoming images in the window. Whenever the user presses a digit on the keyboard ('0'-'9'), the latest image is saved to a folder named with the digit.
 
1. `cd` into the `esp32` folder and build, flash and run the application.
    ```
    idf.py build
    idf.py flash monitor
    ```
    You should see output ending with:
    ```
    I (1262) CAMERA: Camera initialized: 320x240 RGB565.
    Send 'S' to start.
    ```
2. Stop the monitor (`Ctrl/Command+[` or similar) to release the serial port. NOTE: Do not send 'S' in the terminal - the application will start dumping image data into your terminal, and you might have trouble stopping the monitor. The Python application will send the 'S' to start the capture.
3. Open the `python` folder in your Python IDE, select your virtual environment for it, and install dependencies from `requirements.txt`.
4. Run the Python application:
    ```
    python main.py --port <port> --output-path <output-path>
    ```
    * For Windows, <port> should be `COM3` or similar.
    * For Mac and Linux, <port> should be `/dev/ttyACM0` or similar.
    You should see a small window with the camera feed.
5. Start capturing images according to your data collection plan.
