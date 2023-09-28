# video_to_image_and_object_dimension_detection
Converting video to image (assembling object from video frames) and analysing the dimensions of the assembled object based on real reference.

1. python_script:
In the "python_script" directory main python script (vidoe_to_impages_py_37_qt5.py) and qt-designer based interface script (mp4_to_jpg_gui_mask_qt5.ui) is available.
If interface script is modified it has to be compiled by: "pyuic5 -x mp4_to_jpg_gui_mask_qt5.ui -o mp4_to_jpg_gui_mask_qt5.py" in order to get modifications compiled and importable into main python script.


3. windows_runnable:

In the "windows_runnable" directory zip file "video_to_images_and_leaf_dimenstions_WinRunnable.7z" when unzipped a directory containing windows runnable is available.

<pre>
.
├── video_to_images_and_leaf_dimenstions_WinRunnable
    └── video_to_impages_py_37_qt5.exe

After executing "video_to_impages_py_37_qt5.exe" file structure is as following:
.
└── video_to_images_and_leaf_dimenstions_WinRunnable
    ├── video_to_impages_py_37_qt5.exe
    ├── cv2
    ├── ...   
    ├── Outputs
    │   ├── Images and txt output files
    │   └── ...	
    ├── IntermediateOutputs
    │   ├── Video frames ...
    │   └── ...	
    └── Backupimages
        ├── Resulting images outputs ...
        └── ...
</pre>
