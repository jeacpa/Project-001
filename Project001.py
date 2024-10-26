import cv2 as cv
import numpy as np

cap = cv.VideoCapture("Cashmere.MP4")
# cap2 = cv.VideoCapture(0)
framecount = cap.get(cv.CAP_PROP_FRAME_COUNT)
print("frames:",framecount)
# cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
vfname = "vidout.mp4"
# codec = cv.VideoWriter_fourcc('M', 'P', '4', 'V')
codec = cv.VideoWriter_fourcc(*'mp4v')


framerate = 29
resolution = (1920, 1080)

videoOut = cv.VideoWriter(vfname, codec, framerate, resolution)
#ut = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#  LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
def findObjects(outputs, img):

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while cap.isOpened():
    success, img = cap.read()
    if success is not True:
        break
    # imgsize = list(img.shape)
    # print(imgsize)

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv.imshow('Image', img)
    videoOut.write(img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("quitting")
        break
    #cv.waitKey(1)

videoOut.release()
cap.release()
cv.destroyAllWindows()
print("pretty cool!")

print()