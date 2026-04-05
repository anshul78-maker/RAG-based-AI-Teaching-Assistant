import os
import subprocess

files = os.listdir("vedios")
for file in files:
    tutorial_number = file.split(".")[0][1]
    file_name = file.split(".mp4")[0] 
    print(tutorial_number,file_name)
    subprocess.run(["ffmpeg","-i",f"vedios/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])


