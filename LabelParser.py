import json
d = []
with open('/Users/jiyan/Downloads/train/digitStruct.json') as data_file:
    data = json.load(data_file)
    for i in data:
        seqLen = 9
        f = i['filename']
        seq = "#"
        for ii in i['boxes']:
            seqLen -= 1
            lbl = ii['label']
            if lbl == 10.0:
                lbl = 0
            seq = seq + str(int(lbl))
        while seqLen > 0:
            seq = seq + "*"
            seqLen -= 1
        d.append((f, seq))

thefile = open('labels.out', 'w')
for item in d:
    thefile.write("%s,%s\n" % (item[0], item[1]))
thefile.close()