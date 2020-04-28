### Videos to Images
import os
import sys
import subprocess

def convert_video_to_img():
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  for file_name in os.listdir(dir_path):
    if '.mp4' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    counter = name.count("_")
    if counter < 2:
      name = 'real_'+name
    dst_directory_path = os.path.join(dst_dir_path, name)

    video_file_path = os.path.join(dir_path, file_name)
    try:
      if os.path.exists(dst_directory_path):
        
        if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
          subprocess.call('rm -r {}'.format(dst_directory_path), shell=True)
          print('remove {}'.format(dst_directory_path))
          os.mkdir(dst_directory_path)
        else:
          continue
      else:
        os.mkdir(dst_directory_path)
    except:
      print(dst_directory_path)
      continue
    # cmd = 'ffmpeg -i {} -vf scale=-1:360 {}/image_%05d.jpg'.format(video_file_path, dst_directory_path)
    cmd = 'ffmpeg -i {} -vf thumbnail=10,setpts=N/TB -r 1 -vframes 10 {}/image_%05d.jpg'.format(video_file_path, dst_directory_path)
    subprocess.call(cmd, shell=True)
    print("Formated "+video_file_path)
    print('\n')
    frame_len = len(os.listdir(dst_directory_path))
    if frame_len != 10:
      with open('/home/shreshtha/UM/sem2/EECS504/dataset/failure.txt', 'a+') as f:
        f.write(dst_directory_path+" len:"+str(frame_len)+'\n')

###
def check_img_frame_nos():
  dst_dir_path = sys.argv[2]
  for file_name in os.listdir(dst_dir_path):
    data_path = os.path.join(dst_dir_path,file_name)
    len_num = len(os.listdir(data_path))
    if len_num != 10:
      print(data_path)




if __name__=="__main__":
    convert_video_to_img()
    #check_img_frame_nos()
    



