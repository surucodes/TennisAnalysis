from ultralytics import YOLO  

# model = YOLO('yolov8x.pt')

# model = YOLO('models/yolo5_last.pt')
# this trains the model only on the tennis ball(better) and not on the players. thats because we trained yolo5 on tennis ball dataset

# So we need to have 2 passes, one for the players and another for the tennis ball separately. 
model = YOLO('yolov8x.pt')

# The predict function returns a list of ultralytics.engine.results.Results objects — one for each frame (in case of video) or just one if it's an image.
# So if you're running this on a video:

# result[0]
# is the result for the first frame of the video.

# result = model.predict('input_videos/input_video.mp4' , save= True)


result = model.track('input_videos/input_video.mp4' ,conf=0.2,  save= True)
print(result)
# uses botsort by default
# Each result[n] is a Results object, and it has a .boxes attribute, which contains all the bounding boxes detected in that frame.

# More specifically:

# 👉 result[0].boxes is a Boxes object
# It has details like:

# .xyxy: tensor of box coordinates (x1, y1, x2, y2)

# .conf: confidence scores

# .cls: class indices (e.g., 0 for person, 2 for car, etc.)

print("boxes:")
for box in result[0].boxes :
    print(box)
    
# To detect the model better , we finetune our yolo model on tennis ball that is moving fast. We download a custom dataset for the tennis ball from  roboflow and now we fine tune our existing model : 
