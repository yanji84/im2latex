from os import listdir
outfile = open('digit.out','w')

files = listdir("/Users/jiyan/Desktop/class")
for f in files:
  parts = f.split(".")
  label = parts[3][0]
  outfile.write(f + "," + label + "\n")

outfile.close()