import numpy as np
import imutils
import time
import cv2
import os

iconfidence=0.5
ithreshold=0.3

labelsPath=os.path.sep.join(["yolo-coco","coco.names"])
LABELS=open(labelsPath).read().strip().split("\n")
#strip->remove spaces , split->nextline= new object
np.random.seed(42)
COLORS=np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")
#code to form a colored rectangle, 0-255 for 256 colors, 3 for r,g,b channels
weightsPath=os.path.sep.join(["yolo-coco","yolov3.weights"])
#weight-> priority factor of a path from A to B
configPath=os.path.sep.join(["yolo-coco","yolov3.cfg"])
#config->logic/algorithm for the processing of a function/program

net= cv2.dnn.readNetFromDarknet(configPath,weightsPath)
#net->model
ln=net.getLayerNames()
#ln->layer (label name)
ln=[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
#OutLayers-> unnecessary instances outside of required ones must be removed
vc=cv2.VideoCapture(0)
#vc->videocapture from 0->webcam;1->usb cam ,etc.
(W,H)=(None,None)
if imutils.is_cv2():
    prop=cv2.cv.CV_CAP_PROP_FRAME_COUNT
    #for framce count, and ot push next frame over previous one
while True:
    (grabbed,frame) = vc.read()
    #grabbed image is pushed over frame

    if not grabbed:
        break
    
    if W is None or H is None:
        (H,W)=frame.shape[:2]
    
    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    #blob-> chunk of data; swapRB-> swap red and blue(just how it works); crop-> crop image
    start=time.time()
    layerOutputs=net.forward(ln)
    end=time.time()
    #for displaying info regarding time taken for processing

    boxes=[]
    confidences=[]
    classIDs=[]

    for output in layerOutputs:
        for detection in output:

            scores=detection[5:]
            classID=np.argmax(scores)
            confidence=scores[classID]
            #code to parse( extract meaningful data) onto confidence, so detection >50% gets passed on to be displayed
            if confidence > iconfidence:
                box=detection[0:4]*np.array([W,H,W,H])
                (centerX,centerY,width,height)=box.astype('int')
                #code to display box
                x=int(centerX-(width/2))
                y=int(centerY-(height/2))
                #box edge/origin positioning
                boxes.append([x,y,int(width),int (height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                #arrays filled with values(that's what append does) derived from the previous lines of code
    idxs=cv2.dnn.NMSBoxes(boxes,confidences,iconfidence,ithreshold)
    #NMS-> Non Minimal Suppression,boxes stored in idxs
    if len(idxs)>0:
        for i in idxs.flatten:
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])
            #data for origin and corner of rectangle

            color=[int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            text="{}:{:.4f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    cv2.imshow("My App",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


