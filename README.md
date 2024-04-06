# Fall Detection Robot

### Introduction
- Forked from RizwanMunawar/yolov7-pose-estimation
- Combined with fall detection algorithm from manirajanvn/yolov7_keypoints

### Steps to run Code
- Clone the repository.

- Goto the cloned folder.
- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
- Install ffmpeg
- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {Fall-Detection-Robot}


- Run the code with mentioned command below.
```
python main.py

#if you want to change source file
python main.py --source "your custom video.mp4"

#For CPU
python main.py --source "your custom video.mp4" --device cpu

#For GPU
python main.py --source "your custom video.mp4" --device 0

#For View-Image
python main.py --source "your custom video.mp4" --device 0 --view-img

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python main.py --source "your IP Camera Stream URL" --device 0 --view-img

#For WebCam
python main.py --source 0 --view-img

#For External Camera
python main.py --source 1 --view-img
```

- Output file will be created in the working directory with name <b>["your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### References
- https://github.com/WongKinYiu/yolov7
- https://github.com/RizwanMunawar/yolov7-pose-estimation
- https://github.com/manirajanvn/yolov7_keypoints/tree/main

