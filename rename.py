import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is a .jpg file
        if filename.lower().endswith('.jpg'):
            # Find the index of the first '.jpg' occurrence
            base_name = filename.split('.jpg')[0]  # Keep everything before the first '.jpg'
            
            # Construct the new filename (adding '.jpg' back at the end)
            new_filename = 'monster' + base_name + '.jpg'

            # Create the full path for the new filename
            new_file_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

# Set the directory where your .jpg files are located
directory = 'CANS-REGRESSION\class4.class'
rename_files_in_directory(directory)
