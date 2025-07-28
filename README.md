### Steps that I have followed to implement this project (in great detail)
* Inside the yolo_inference.py :
    * 1. Make a Virtual Environment and Import ultralytics : It’s like the all in one tool for detection, classification, segmentation, tracking, and training  all with YOLOv8 and beyond.
    * 2. Test the Yolov8 model with a single image. Results: It correctly classified/detected the people , chair with high confidence but the tennis ball wasnt getting detected in all frames upon trying with a video. Conclusion: Either finetune yolov8 or train another yolov5 model on a custom dataset obtained from roboflow and then have two different passes for detection, one with yolov8 for players and another with yolov5 for the tennis ball. -> Going with the latter approach.
    * 3. The bounding box in one frame is currently not matching the other. i.e the objects are not being tracked currently. We test the ultralytics's model.track for now and it is saved in the predict 5 file. Result : Maintains different ID for each different player now and the bounding boxes in consecutive frames remain same track the objects. 
        * 3.1. Ultralytics' model.track() Employs BoT-SORT and ByteTrack Algorithms.
        Ultralytics' powerful model.track() function utilizes two state-of-the-art tracking algorithms: BoT-SORT and ByteTrack. By default, the function employs BoT-SORT for object tracking tasks.

        Users have the flexibility to switch to the ByteTrack algorithm by specifying it in the corresponding YAML configuration file. This allows for adaptability based on the specific requirements of the tracking scenario.

        The track() method is designed to be a seamless extension of the object detection capabilities, enabling users to not only detect objects in a video stream but also to assign and maintain a unique ID for each detected object across consecutive frames. This functionality is crucial for a wide range of applications, including video surveillance, traffic analysis, and sports analytics.
        
        * 3.2. As we have only a single ball over here, Theres no need to track the ball in this case, only detection is enough. 
    * 4. We now need to train the model to detect keypoints on the tennis court. For that , we download a dataset using wget containing many images and json files having their 14 keypoint location's x and y coordinates. The json file has an id : associated with the image, has a ,list of 14 x and y data points and also 
    * 5.
    * 6.
    * 7.
    * 8.
    * 9.

