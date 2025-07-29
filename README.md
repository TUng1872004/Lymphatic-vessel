Segmentation Demo
=================

This is an image segmentation demo built with Python. It uses PyTorch and segmentation-models-pytorch for the model, and tkinter for the graphical user interface.

Requirements
------------
Install all required dependencies with:

    pip install -r requirements.txt

How to Run
----------
To launch the demo, run:

    python Demo.py

Features
--------
- Load images via file dialog
- Apply pretrained segmentation models
- Display results using matplotlib inside the GUI

Files
-----
- Demo.py: Main application script
- requirements.txt: List of Python dependencies

Notes
-----
- tkinter is usually bundled with Python. On Linux systems, you may need to install it manually:

      sudo apt-get install python3-tk

- For better performance, ensure that you have a GPU and a CUDA-enabled version of PyTorch.

Enjoy segmenting!
