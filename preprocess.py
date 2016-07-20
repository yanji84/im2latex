import csv
import cv2

source_path = '/Users/jiyan/Downloads/InftyCDB-2/English/Images_eng/'
dest_path = '/Users/jiyan/Downloads/InftyCDB-2/English/processed/'

f = open('/Users/jiyan/Downloads/InftyCDB-2/English/Characters_eng.csv')
csv_f = csv.reader(f)

index = 0
for row in csv_f:
  label = row[5]
  source_filename = row[15]
  coords = (int(row[16]), int(row[17]), int(row[18]), int(row[19]))
  dest_filename = str(index) + '.png'
  index += 1

  source_file_path = source_path + source_filename
  img = cv2.imread(source_file_path,0)
  img = img[coords[1]:coords[3],coords[0]:coords[2]]
  cv2.imwrite(dest_path + dest_filename, img)

  print label, source_filename, dest_filename, coords
