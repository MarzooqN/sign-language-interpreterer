import cv2
import mediapipe as mp

#get holistic model (ability to detect things) and drawing(being able to "draw" on the feed whats actually being tracked)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):

    #Converts from BGR to RGB (cv2 default BGR mediapipe need RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #image is no longer writeable 
    image.flags.writeable = False

    #make prediction 
    results = model.process(image)

    #image is now  writeable 
    image.flags.writeable = True

    #convert image back
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
   
    #Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                              ) 
     
    #Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    
    #Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    
    #Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 


def run_camera_no_NN():
    #Open camera
    cap = cv2.VideoCapture(0)

    #Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
        while cap.isOpened():

            #Get frame
            ret, frame = cap.read()

            #make detection 
            image, results = mediapipe_detection(frame, holistic)
            
            #Draw landmarks
            draw_landmarks(image, results)

            #Show frame
            cv2.imshow('Camera Feed', image)

            #If press q then break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        #Destory all windows  
        cap.release()
        cv2.destroyAllWindows()


