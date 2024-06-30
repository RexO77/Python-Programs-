import os 
from zipfile import ZipFile
with ZipFile("path.zip")as obj:
    for folder_name,sub_folder,file_name in os.walk("path.zip"):
        for filename in file_name:
            file_path = os.path.join(folder_name,filename)
            obj.write(file_path,os.path.basename(file_path))
if os.path.exists("same file path.zip"):
    print("Zip created ")
else:
    print("Zip not created")