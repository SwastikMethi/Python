import os
import shutil

def search_files_and_copy(directory, keywords, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    for root, dirs, files in os.walk(directory):
        if files:
            for file_name in files:
                if file_name.endswith('.txt'):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if any(keyword in content for keyword in keywords):
                            print(f"Found keywords in: {file_path}")
                            shutil.copy(file_path, destination_directory)
        else:
            print("No text files found in the specified directory.")

directory_path = r'directory_path'  # Specify the directory path here
keywords = ['keyword'] # Specify the keywords here
destination_path = r'destination_path' # Specify the destination path here 

search_files_and_copy(directory_path, keywords, destination_path)
