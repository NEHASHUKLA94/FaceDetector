import cv2 #this library is used to open deal with video image #this is the most popular library used in the world to deal with video and image processing
import dlib #this is used for face detetctor model, why?- haarcascade model can be used too , comes in file, this is an alternative so that we can directly use from library or else we would have to dl haarcascade file
import imutils # thisis used for resizing alone the frame


#this is used to fix the coordinates of faces  
def convert_and_trim_bb(image, rect):
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	w = endX - startX
	h = endY - startY
	return (startX, startY, w, h)

#we are loading the dlib the facedetector model using this frontal function
faceDetector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)# this is used for open camera with parameter 0 but if we pass a video file path, it will open that file
while True:
    _, frame = cam.read() #_ variable is used for status, we dont need it here
    frame = imutils.resize(frame, width=640)
    faces = faceDetector(frame)
    if(len(faces) > 0):
        cv2.putText(img=frame, text="Hi human being", org=(5,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
    else:
       cv2.putText(img=frame, text="Not A Human Being", org=(5,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
    boxes = [convert_and_trim_bb(frame, r) for r in faces] 
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)    
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()