'''
File helper for moving/renaming/etc image training data.

Syntax:
rename.py <input_dir> <output_dir> <mode> [<parameter>]

Mode can be:
split (test/train split) with parameter train ratio
batch (break images into groups) with parameter batch size

Written for .jpg only.  Should work for others with minor changes

Copyright (c) 2025 Tillman. All Rights Reserved.
'''

import os
import random
import sys

# params to use if no arguments
MODE = 'split'
directory = './dir1'  # input directory (or directory for rename)
output_dir = './dir2'

# if args: let args control
if len(sys.argv) > 1:
    directory = sys.argv[1]
    output_dir = sys.argv[2]
    MODE = sys.argv[3]
    if sys.argv[3] == 'rename':
        print('ERROR: argument based renaming not supported.  Modify script to use this mode.')
        exit()


def main():
    errors = False
    if MODE == 'rename':
        rename(directory)
    elif MODE == 'split':
        try:
            split(directory, output_dir, float(sys.argv[4]))
        except:
            print('WARN: no split ratio, default: 0.8')
            split(directory, output_dir, 0.8)
            errors = True
    elif MODE == 'batch':
        try:
            batch(directory, output_dir, int(sys.argv[4]))
        except:
            print('WARN: no split ratio, default: 100')
            batch(directory, output_dir, 100)
            errors = True
    else:
        print('WARN: arguments detected but no mode specified.')
        errors = True
    
    if errors:
        print('\nJob completed with errors.  See WARN above\n')
    else:
        print('\nJob completed, no errors reported\n')

def rename(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith('.jpg'):
            base_name = filename.split('.jpg')[0]  # Keeps everything before the first '.jpg'
            
            # Construct the new filename, change as nessesary but keep .jpg!
            new_filename = '2' + 'redbull' + base_name + '.jpg'
            new_file_path = os.path.join(directory, new_filename)

            os.rename(file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

def copy_file(src, dest):
    with open(src, 'rb') as fsrc:
        with open(dest, 'wb') as fdest:
            fdest.write(fsrc.read())

def split(directory, output_dir, train_ratio):
    # splits images randomly into train and test subdirectories in output_dir.
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    others = [f for f in os.listdir(directory) if not f.lower().endswith('.jpg')]
    images = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images, test_images = images[:split_index], images[split_index:]

    for img in train_images:
        src_path = os.path.join(directory, img)
        dest_path = os.path.join(train_dir, img)
        copy_file(src_path, dest_path)

    for img in test_images:
        src_path = os.path.join(directory, img)
        dest_path = os.path.join(test_dir, img)
        copy_file(src_path, dest_path)

    # Keeps non-.jpg files in parent folder
    for nonimg in others:
        src_path = os.path.join(directory, nonimg)
        dest_path = os.path.join(output_dir, nonimg)
        copy_file(src_path, dest_path)

    print(f"Copied {len(train_images)} images into {train_dir}")
    print(f"Copied {len(test_images)} images into {test_dir}")

def batch(directory, output_dir, batch_size):
    # Creates subdirectories with batch_size images each.
    images = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    random.shuffle(images)

    batch_num = 1
    for i in range(0, len(images), batch_size):
        # Create the batch directory inside the output_dir
        batch_dir = os.path.join(output_dir, f'batch_{batch_num}')
        os.makedirs(batch_dir, exist_ok=True)

        for img in images[i:i + batch_size]:
            src_path = os.path.join(directory, img)
            dest_path = os.path.join(batch_dir, img)
            copy_file(src_path, dest_path)

        print(f"Created {batch_dir} with {len(images[i:i + batch_size])} images")
        batch_num += 1

if __name__ == '__main__':
    main()