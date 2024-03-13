import cv2
import numpy as np
import os
import mediapipe as mp
from main import mediapipe_detection, draw_landmarks, mp_holistic
import json
import urllib.request
from pytube import YouTube
from cap_from_youtube import cap_from_youtube


class Data:

    def __init__(self, MS: bool):
       
        #30 videos worth of data for each actions (mentioned above)
        self.no_sequences = 30

        #each video is going to be 30 frames instead of just 1
        self.sequences_length = 30
       
        if MS:
            # Update actions based on MSASL_classes.json
            with open('MS-ASL/MSASL_classes.json', 'r') as f:
                
                #For testing
                data_set = json.load(f)
                
                #Non-testing
                '''
                self.actions = np.array(json.load(f))
                '''

            selected_words = ['gym', 'my', 'name', 'talk' , 'leave', 'so', 'vegetable', 'next week', 'interpreter', 'weekend', 'go', 'shower', 'week', 'beard' ]
            
            filtered_dataset = [word for word in data_set if word in selected_words]

            with open('filtered_dataset.json', 'w') as f:
                json.dump(filtered_dataset, f)

            with open('filtered_dataset.json', 'r') as f:
                self.actions = np.array(json.load(f))
            
            # Update DATA_PATH
            self.DATA_PATH = os.path.join('MSASL_Data')

            #Testing path
            self.TESTING_PATH = os.path.join('MSASL_Test_Data')

            if(not(os.path.exists('MSASL_Test_Data'))):
                os.makedirs(self.TESTING_PATH)

            #Actual
            '''
            if(not(os.path.exists('MSASL_Data'))):
                # for action in self.actions:
                os.makedirs(self.DATA_PATH)
            '''

        else:
            #Path for exported data, numpy arrays
            self.DATA_PATH = os.path.join('MP_Data')

            #Actions to detect (add others actions to detect other actions)
            self.actions = np.array(['hello', 'thanks', 'iloveyou'])

            if(os.path.exists('MP_Data')):
                self.create_folders()

        


    def create_folders(self):
        for action in self.actions:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(self.DATA_PATH,action, str(sequence)))
                except:
                    pass



def extract_keypoints(results):
   
    #Creates arrays of numbers for different landmarks, if None then will create an array of 0's 
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) #21 * 3 because there are 21 landmarks each with 3 coordinate values
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])


def data_collections(Data):
    #Open camera
    cap = cv2.VideoCapture(0)

    #Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
        
        #Loop through actions 
        for action in Data.actions:

            #Loop through sequeneces aka videos
            for sequence in range(Data.no_sequences):

                #Loop through video length aka sequence length
                for frame_num in range(Data.sequences_length):
            
                    #Get frame
                    ret, frame = cap.read()

                    #make detection 
                    image, results = mediapipe_detection(frame, holistic)
                    
                    #Draw landmarks
                    draw_landmarks(image, results)

                    #Apply wait logic 
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 1, cv2.LINE_AA)
                        
                        #Show to Screen
                        cv2.imshow('Camera Feed', image)
                        cv2.waitKey(1500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        #Show to Screen
                        cv2.imshow('Camera Feed', image)

                    #Extract keypoints and save them into designated folder
                    keypoints = extract_keypoints(results)
                    npy_path=os.path.join(Data.DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)


                    #If press q then break
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        #Destory all windows  
        cap.release()
        cv2.destroyAllWindows()


def download_video(url, output_path):
    yt = YouTube(url)
    video = yt.streams.first()
    video.download(output_path)


def collect_data_MS(Data: Data):

    for split in ['train', 'test', 'val']:
        with open(f'MS-ASL/MSASL_{split}.json', 'r') as f:
            dataset = json.load(f)

        for sample in dataset:
            # Download video if not already downloaded
            video_url = sample['url']
            # video_filename = f"{sample['file']}.mp4"
            # video_path = os.path.join(Data.VIDEO_PATH, 'videos', video_filename)

            # if not os.path.exists(video_path):
            #     download_video(video_url, video_path)

            # Load video

            try:
                video = cap_from_youtube(video_url) 
            except:
                continue


            os.makedirs(os.path.join(Data.DATA_PATH, str(sample['text'])), exist_ok=True)

            folder_path = os.path.join(Data.DATA_PATH, sample['text'])

            # Count the number of subdirectories in the specified folder
            video_num = sum(1 for _ in os.walk(folder_path)) - 1

            os.makedirs(os.path.join(Data.DATA_PATH, sample['text'], f"{video_num}" ))

            # Set mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
                
                #Get the total number of frames in the video 
                video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                #Calculate the intervals for which the video will analyze which frame
                skip_frames_window = max(int(video_frame_count/Data.sequences_length), 1)

                #Iterate through the frames
                for frame_num in range(Data.sequences_length):
                    
                    #Set the urrent frame to the position needed
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num * skip_frames_window)
                    
                    #Read the frame from the video
                    ret, frame = video.read()

                    #Error handling: if the video is corrupt break the loop
                    if not ret:
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    

                    # draw_landmarks(image, results)


                    # Extract keypoints and save them into designated folder
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(Data.DATA_PATH, str(sample['text']), str(video_num), str(frame_num))
                    np.save(npy_path, keypoints)

                    # If press q then break
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            video.release()
            cv2.destroyAllWindows()


def collect_data_MS_test(Data: Data):

    for split in ['train', 'test', 'val']:
        with open(f'MS-ASL/MSASL_{split}.json', 'r') as f:
            dataset = json.load(f)

        for sample in dataset:
            if sample['text'] in Data.actions:
  
                video_url = sample['url']

                try:
                    video = cap_from_youtube(video_url) 
                except:
                    continue


                os.makedirs(os.path.join(Data.TESTING_PATH, str(sample['text'])), exist_ok=True)

                folder_path = os.path.join(Data.TESTING_PATH, sample['text'])

                # Count the number of subdirectories in the specified folder
                video_num = sum(1 for _ in os.walk(folder_path)) - 1

                os.makedirs(os.path.join(Data.TESTING_PATH, sample['text'], f"{video_num}" ))

                # Set mediapipe model
                with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
                    
                    #Get the total number of frames in the video 
                    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                    #Calculate the intervals for which the video will analyze which frame
                    skip_frames_window = max(int(video_frame_count/Data.sequences_length), 1)

                    #Iterate through the frames
                    for frame_num in range(Data.sequences_length):
                        
                        #Set the urrent frame to the position needed
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num * skip_frames_window)
                        
                        #Read the frame from the video
                        ret, frame = video.read()

                        #Error handling: if the video is corrupt break the loop
                        if not ret:
                            break

                        image, results = mediapipe_detection(frame, holistic)
                        

                        # draw_landmarks(image, results)


                        # Extract keypoints and save them into designated folder
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(Data.TESTING_PATH, str(sample['text']), str(video_num), str(frame_num))
                        np.save(npy_path, keypoints)

                        # If press q then break
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                video.release()
                cv2.destroyAllWindows()


