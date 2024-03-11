from main import mp_holistic, mediapipe_detection, draw_landmarks
from collect_data import extract_keypoints
import cv2
import numpy as np
from neural_network import create_model
from collect_data import Data



def run_real_time(Data: Data):
    #Load Model
    model = create_model()


    #Detection Variables
    sequence = []
    sentence = []
    threshold = 0.4

    cap = cv2.VideoCapture(0)

    #Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
        while cap.isOpened():

            #Get frame
            ret, frame = cap.read()

            #make detection 
            image, results = mediapipe_detection(frame, holistic)
            
            #Draw landmarks
            # draw_landmarks(image, results)

            #Prediction logic 

            #Get keypoints, appened them to the sequence then get the last 30 frames
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            #Flip image to see properly
            image = cv2.flip(image, 1)


            if len(sequence) == 30:

                #Use expand dims for proper formatting for np array (allows to pass through 1 sequence at a time)
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                prediction_pos = np.argmax(res)

                #Visualization 

                #if the predicted value is above a .4 then check if there is anything in the sentence, if the last word is the same word as before dont do anything
                if res[prediction_pos] > threshold:
                    if len(sentence) > 0:
                        if Data.actions[prediction_pos] != sentence[-1]:
                            sentence.append(Data.actions[prediction_pos])
                    else:
                        sentence.append(Data.actions[prediction_pos])

                #if there are more than 5 words only grab last 5 words
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                #Puts sentence on the screen
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            
            #Show frame
            cv2.imshow('Camera Feed', image)

            #If press q then break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        #Destory all windows  
        cap.release()
        cv2.destroyAllWindows()