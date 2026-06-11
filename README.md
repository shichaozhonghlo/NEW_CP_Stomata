User Manual for the Catalpa bungei Leaf Stomata Detection System Based on YOLO11-ESC

1 Introduction

This document provides a clear and complete guide for the Catalpa bungei stomatal phenotyping system. Its objectives are as follows. First, it clarifies system functions and operational procedures by describing the software features, interface, and stepwise operations to ensure that users can quickly learn to load images, perform automated detection, analyse results, and manage data accurately. Second, it defines software capabilities and standards, serving as a reference for design, development, and acceptance criteria, thereby ensuring consistency between software outcomes and user requirements. Third, it provides technical references for maintenance personnel, testers, and potential future developers to understand software architecture and interaction logic.

The primary audience for this document includes end users such as researchers and technicians engaged in plant physiology, breeding, or phenomics, as well as project managers, software testers, and technical support staff responsible for system maintenance.

2 System Operation

2.1 System Structure

2.1.1 Software Architecture

The system follows a modular architecture with the following directory structure

stomata_gui/
│
├─ main.py
│
├─ gui/
│ ├─ main_window.py
│ ├─ home_page.py
│ └─ image_detect_page.py
│
└─ core/
  └─ stomata_detector.py

2.1.2 Home Page

The home page contains several functional options. At the top, the interface provides home and image detection options. At the bottom, it allows model switching and access to historical results. Each option navigates to its respective interface.

2.1.3 Image Detection

Selecting the image detection option opens an interface to load stomatal images and perform detection. Users first select images to upload. Once uploaded, the start detection option executes the detection workflow. The resulting detected images appear in the result display section.

2.1.4 Model Switching

The model switching interface allows users to load the best.pt file containing the optimal YOLO11 trained weights for detection.

2.1.5 Viewing Detection Results

The detection results section organizes output into two folders. The img folder stores images annotated with detected stomatal regions, while the txt folder contains text files recording the detailed coordinates and confidence scores of each detection.

2.2 Usage Instructions

2.2.1 Running the Main File

The main.py file can be executed in PyCharm. Running this file launches the application interface. The core code instantiates the main window, shows it, and executes the event loop.

2.2.2 Home Interface

After launching the main file, the home interface appears, providing navigation to other functional modules.

2.2.3 Switching the Detection Model

Users select the best trained weight file produced by YOLO11 and confirm the path display, indicating the model parameters have been successfully loaded.

2.2.4 Image Input

After loading the model weights, users select the image detection option and choose JPG format stomatal images for analysis. Loaded images appear in the designated input section.

2.2.5 Stomatal Detection

Executing the start detection option runs the detection algorithm. Completion is confirmed when the interface displays a completion message.

2.2.6 Viewing Detection Results

Returning to the home interface allows users to access the detection results. Users can inspect annotated images in the img folder and view detailed detection data in the txt folder.





