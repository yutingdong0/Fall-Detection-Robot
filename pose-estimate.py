import math
import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        person_fell_down = False  # Flag to track if person fell down

        while(cap.isOpened): #loop until cap opened or video not complete
            
            # Flag to track whether the person fall down in this frame
            condition_to_detect_fall_down = False

            # print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                # Fall detection part
                # Apply non max suppression
                # output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                # output = output_to_keypoint(output)
                image0 = image[0].permute(1, 2, 0) * 255
                image0 = image0.cpu().numpy().astype(np.uint8)
        
                #reshape image format to (BGR)
                image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
                for idx in range(output.shape[0]):
                    #plot_skeleton_kpts(image0, output[idx, 7:].T, 3)
                    xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                    xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

                    left_shoulder_y= output[idx][23]
                    left_shoulder_x= output[idx][22]
                    right_shoulder_y= output[idx][26]
            
                    left_body_y = output[idx][41]
                    left_body_x = output[idx][40]
                    right_body_y = output[idx][44]

                    len_factor = math.sqrt(((left_shoulder_y - left_body_y)**2 + (left_shoulder_x - left_body_x)**2 ))

                    left_foot_y = output[idx][53]
                    right_foot_y = output[idx][56]
            
                    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
                    #Plotting key points on Image
                        cv2.rectangle(image0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(0, 0, 255),
                        thickness=5,lineType=cv2.LINE_AA)
                        cv2.putText(image0, 'Person fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
                        condition_to_detect_fall_down = True
                    else:
                        cv2.putText(image0, 'Person not fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
                        condition_to_detect_fall_down = False

                    if condition_to_detect_fall_down:
                        if not person_fell_down:  # Check if person hasn't fallen down already
                        # Person fell down, so use NotifyRecordParse class
                            notify = NotifyRecordParse()
                            if notify.activate():
                                print("Call needed!") 
                            else:
                                print("Call not needed.")  
                                person_fell_down = True  # Set flag to indicate person fell down
                    else:
                        person_fell_down = False  # Reset flag if person is not detected to have fallen down



                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            # print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                        orig_shape=im0.shape[:2])

                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", image0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(image0)  #writing the video frame

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
