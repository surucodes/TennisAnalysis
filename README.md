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

    * 4. We now need to train the model to detect keypoints on the tennis court. For that , we download a dataset using wget containing many images and json files having their 14 keypoint location's x and y coordinates. The json file has an id : associated with the image, has a list of 14 x and y data points and also a metric score of the keypoint.

        * 4.1. We create a class called keypoint detector, which inherits form the Dataset calss of torch , so that it can be used seamlelssly in the data loader later. Now we transform all the images using transformer provided by pytorch where we resize , normalise and also convert it into a pil and tensor format.  We include a method called get item where we return the image and its respective coordinates from the json file. It is important to note that as we transform the images, its keypoint coordinates also need to be respectively changed according to the new image size. We do the same through simple mathematical calculation for each image. The KeypointsDataset class is a custom PyTorch Dataset. Its purpose is to define how individual image-keypoint pairs are loaded, preprocessed, and made available for the DataLoader.

        * 4.2. We now send the individual train and validation images to the keypoint detector class and then get the Dataset class. The DataLoader is a utility that wraps a Dataset and provides an efficient way to iterate over the data in batches. Role of DataLoader:
        Batching: Collects multiple individual items returned by Dataset.__getitem__ into a single batch (a larger tensor) and shuffles them. 

        This is what happens conceptually in the first iteration (and subsequent iterations):

        Request for a Batch: The DataLoader needs a batch of 8 (img, kps) pairs.

        Calls Dataset.__getitem__ 8 Times: The DataLoader (or its worker processes if num_workers > 0) will internally call train_dataset.__getitem__(idx) eight different times, each time with a different idx (chosen randomly if shuffle=True).

        It calls train_dataset.__getitem__(0), gets (img_0, kps_0).

        It calls train_dataset.__getitem__(17), gets (img_17, kps_17).

        ... (and so on, 6 more times with random indices) ...

        It calls train_dataset.__getitem__(92), gets (img_92, kps_92).

        Each of these individual (img, kps) pairs has gone through the full KeypointsDataset.__getitem__ pipeline: image loading, resizing, scaling to [0,1], normalizing, and keypoint scaling.

        Collates into a Batch Tensor: Once it has these 8 individual (image_tensor, keypoints_numpy_array) pairs, the DataLoader's default collate_fn (which works well for tensors and NumPy arrays of compatible shapes) combines them:

        All 8 image_tensors (each of shape (3, 224, 224)) are stacked along a new dimension to form a single batch tensor for images. Its shape will be (8, 3, 224, 224).

        All 8 keypoints_numpy_arrays (each with a shape like (6,) if you have 3 keypoints, (14*2,) if you have 14 keypoints) are stacked to form a single batch tensor for keypoints. Its shape will be (8, 6) or (8, 14*2).

        Yields the Batch: The DataLoader then yields these two batch tensors:

        images (a torch.Tensor of shape (8, 3, 224, 224))

        keypoints_batch (a torch.Tensor of shape (8, N), where N is the total number of keypoint coordinates). Note: The np.array from __getitem__ is automatically converted to torch.Tensor by the DataLoader's default collate_fn.

        * 4.3. we now train a transfer learning model resnet50 by freezing its previous layers and only changing the final layer for it to learn the keypoint extraction from the given dataset.The original ResNet50 model's final fully connected layer (model.fc) was designed to output 1000 classes (for ImageNet classification).
        Here, we replace this final layer with a new one tailored for our
        specific task of keypoint detection.
        'model.fc.in_features' ensures that the new layer receives the
        correct number of input features from the preceding ResNet50 layers,
        preserving the feature extraction capabilities of the pre-trained backbone.
        '14 * 2' specifies the number of output units for our new task.
        This typically means 14 keypoints, where each keypoint has an X and a Y coordinate,
        resulting in 28 output values (e.g., x1, y1, x2, y2, ..., x14, y14).

        * 4.4. We now make use of a MSE loss function to check how much the models output is varying from the actual keypoints. To converge and minimize the loss funcition, we make use of Adam optimizer and feed in the model's parameters along with the learning rate. We then finally train the model like
            optimizer.zero_grad()     # Step 1: Clear previous gradients
            outputs = model(imgs)     # Step 2: Forward pass (get predictions)
            loss = criterion(outputs, kps)  # Step 3: Compute loss
            loss.backward()           # Step 4: Compute gradients
            optimizer.step()          # Step 5: Update weights
        
        * 4.5. We now write a utils file to read the video frames and to also save the video outputs. We use lossy methods for compression of images and saving them here: 
        What is a Codec?
        A codec (short for coder-decoder) is used to compress (encode) and decompress (decode) digital data, usually audio or video.

        🧠 Purpose:
        Encoding: Converts raw data into a compressed format to reduce size and enable storage/transmission.

        Decoding: Converts compressed data back for viewing or playback.

        💡 Think of a codec as a translator that compresses video for storage and decompresses it for viewing.

        🔹 Why Do We Need Codecs?
        🔸 Raw video size example:
        1 frame of 1920×1080 RGB =
        1920 × 1080 × 3 = 6,220,800 bytes ≈ 6 MB

        1-minute video at 24 FPS =
        6 MB × 24 × 60 = ~8.6 GB
        🔥 Too large for practical storage or streaming.

        ✅ Codecs reduce file size (with or without quality loss).
        🔹 Types of Codecs:
        1. Lossy Codecs
        Compress by discarding data less noticeable to humans.

        ✅ Smaller file size

        ❌ Some loss in quality

        🧾 Examples: MJPG, H.264, MP3

        2. Lossless Codecs
        Compress without losing any data.

        ✅ Perfect quality

        ❌ Larger file size

        🧾 Examples: FLAC, H.265 (in certain modes)

        🔹 What is FourCC?
        FourCC = Four Character Code
        A 4-byte identifier that tells software which codec to use.

        📦 In our code:
       
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        'MJPG' = Motion JPEG codec

        *'MJPG' = Python unpacking: 'M', 'J', 'P', 'G'

        cv2.VideoWriter_fourcc() converts those 4 chars into a 4-byte code used by OpenCV

        ✅ cv2.VideoWriter_fourcc(*'MJPG') = cv2.VideoWriter_fourcc('M','J','P','G')

        🔹 What is MJPG (Motion JPEG)?
        A video codec where:

        Each frame is individually compressed as a JPEG image

        🔑 Features:
        🧩 Intra-frame compression
        Each frame is compressed independently
        → Fast, simple; no frame-to-frame comparison

        📦 Larger files than modern codecs like H.264
        (due to lack of inter-frame compression)

        ✅ High compatibility
        Supported across platforms and video tools

        ⚙️ Use cases:
        Surveillance, simple video export, fast saving in OpenCV

        💡 In our code, MJPG is chosen because it’s widely supported and doesn’t rely on complex dependencies.

        * 4.6. We now tese the save funciton by giving the same input video. We can see that the avi format made the original video which was 7 seconds to be now 9 seconds. Hence the avi format is bulky and occupies more size than the jpeg format. 


    * 5. The next step is to detect the players. We define some functions for the same task. We define a class called the PLayerTracker and we have some functions inside this which will help us to get the frames from the video and also the annotations associated with the images. Looking form the top, we have 3 funcitons detect frame , detect frames and draw bounding boxes funcitons. 
        * 5.1. The detect frame function takes in a single frame , applies the default botsort through model.track (with persist=True , to specify that video frames are being inputted) and then stores the result in a 'result' variable. The result.names gives retrieves class ids and class names with a {id :"class"} dictionary like {0 : "person" , etc} . We then access the iterable results.boxes which has all the information related to the detected bounding boxes and we only keep the bounding boxes of the people class. box.id gives us a tensor which is the tracking id of the partiocular person , box.xyxy gives me the bounding box coordinates , box.cls returns a tensor of the class id ,(same as and associated with the results.names). And then we filter out only the person object class and store a dictionary which has player track id as keys and the bounding box coordinates as the values. 

        The detect_frame function:
        1. Takes a single video frame as input.
        2. Uses a pre-trained object detection and tracking model (likely YOLO or similar) to detect and track objects in the frame.
        3. Filters the results to keep only objects classified as "person."
            For each detected person:
            Extracts their tracking ID (a unique identifier that persists across frames).
            Extracts their bounding box coordinates ([x_min, y_min, x_max, y_max]).
            Stores this information in a dictionary (player_dict) where the keys are tracking IDs and the values are bounding box coordinates.
        Returns the dictionary, which can be used to track people across frames in a video (e.g., for analyzing player movements in a sports video).

        * 5.2. The detect frames funciton makes use of the detect frame funciton to input and iterate through all the frames in a video and then store all the frames in a separate list of dictionaries where ever element specifies the parameters of a specific frame with the track ids and the bounding box coordinates of only the players. 

        * 5.3. The draw_bboxes funciton is a funciton to annotate the bounding boxes on each frame. Inside this funciton , we simultaneously access the video frames one by one and also the player detections list of dictionary. Note that length of video frames (which is structured something like video frames = {frame1 , frame 2 ,..} where each frame is a numpy array) is same as the length of player_detections as we have extracted tracking IDs and  coordinates of the bounding boxes for the very same frames. 
        We then iterate through a single frame and a single element from the player detection and then for every frame , access the items in the individial frame's coordinates dictionary and then annotate using cv2.rectangle and also put up a tracking ID text just above each bounding box. We return the video frames with annotated values appended in a single output_video_frames list. 

        * 5.4. We read the video in the video_frames variable using the read_video function and then annotate using draw_bbox function.
      







    * 6.

    * 7.

    * 8.

    * 9.

