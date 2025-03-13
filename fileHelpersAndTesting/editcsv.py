"""
File helper for CSV editing

Change as needed
"""


import csv

def modify_csv(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # modify rows as needed
        for row in reader:
            if row: # if the row exists
                row[0] = '2' + row[0]
            writer.writerow(row)

modify_csv('CANS-REGMASK2/angles.csv', 'output.csv')
