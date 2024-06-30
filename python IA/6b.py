import os
from zipfile import ZipFile
with ZipFile(r'C:\Programming\Python Programs\python IA','w') as zipobj:
    for folder_name,sub_folder,file_name in os.walk(r'C:\Programming\Python Programs\python IA'):
        for filename in file_name:
            file_path = os.path.join(folder_name,filename)
            zipobj.write(file_name,os.path.basename(file_name))
if os.path.exists(r"C:\Programming\Python Programs\python IA.zip"):
    print("Zip FIle created")
else:
    print("Zip file not created")