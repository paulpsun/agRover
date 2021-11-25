import cv2
import os



#Video directory, where to get video
directory = '/media/jetson/ssd/furrow6_video/videos'
    
for folder in os.scandir(directory):
    if folder.is_dir():
        folder_path = folder.path
        print("Class folder path: " + folder_path)
        
        folder_name = folder.name
        print("Class folder name: "+ folder_name)      
        
        for entry in os.scandir(folder_path):
            if entry.path.endswith(".MTS") and entry.is_file():
                
                file_path = entry.path
                print("video file path: "+file_path)
        
                filename = os.path.splitext(entry.name)[0]
                print("video file name: "+filename)
        
                vidcap = cv2.VideoCapture(file_path)
                success,image = vidcap.read()
                count = 0
                
                # Image directory, Where to save
                directory_save = '/media/jetson/ssd/furrow6_video'
                         
                
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



            
          
