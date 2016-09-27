import json

filePath = '/Users/jiyan/Desktop/digitStruct.json'

def parseLabelSeq():
    d = []
    with open(filePath) as data_file:
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

def parseBoundingBox():
    d = []
    with open(filePath) as data_file:
        data = json.load(data_file)
        for i in data:
            f = i['filename']
            t = i['boxes'][0]['top']
            l = i['boxes'][0]['left']
            w = i['boxes'][0]['width']
            h = i['boxes'][0]['height']
            d.append((f, t, l, w, h))

    thefile = open('labels.out', 'w')
    for item in d:
        thefile.write("%s,%s,%s,%s,%s\n" % (item[0], item[1], item[2], item[3], item[4]))
    thefile.close()

parseBoundingBox()