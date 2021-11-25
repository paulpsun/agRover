import cv2
import os


#folder name
folder_name = 'straight'

#Video directory, where to get video
directory = '/media/jetson/Samsung/agbot_data_furrow_5th/labeled_video_folders/'+folder_name
for entry in os.scandir(directory):
    if entry.path.endswith(".MTS") and entry.is_file():
        
        file_path = entry.path
        print(file_path)

        filename = os.path.splitext(entry.name)[0]
        print(filename)

        vidcap = cv2.VideoCapture(file_path)
        success,image = vidcap.read()
        count = 0
        
        # Image directory, Where to save
        directory_save = '/media/jetson/ssd/_training_orient_f5'
                 
        
        while success:
            
          filename_save = filename + '_%d.jpg' % count

          if count % 5 == 0 or count % 5 == 1 or count % 5 == 2:
            cv2.imwrite(directory_save+'/train/'+ folder_name +'/'+ filename_save, image)     # save frame as JPEG file
            
          if count % 5 == 3:
            cv2.imwrite(directory_save+'/val/'+ folder_name +'/'+ filename_save, image)     # save frame as JPEG file  
            
          if count % 5 == 4:
            cv2.imwrite(directory_save+'/test/'+ folder_name +'/'+ filename_save, image)     # save frame as JPEG file      
        
          success,image = vidcap.read()
          
          #print status every 100 frames
          if count % 100 == 0:
              print('Read a new frame: %d' % count, success)
              
          count += 1
