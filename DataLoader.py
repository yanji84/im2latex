import cv2

class DataLoader(object):
    def __init__(self, path, dimension, clasDict, nClasses, nSteps, debug=False):
        print "loading raw data..."
        self.train_idx = 0
        self.train_images = []
        self.train_x = []
        self.train_y = []
        self.train_mask = []

        self.test_idx = 0
        self.test_images = []
        self.test_x = []
        self.test_y = []
        self.test_mask = []

        self.read_file(path, "train.out", self.train_images, self.train_x, self.train_y, self.train_mask, dimension, clasDict, nClasses, nSteps, debug)
        self.read_file(path, "test.out", self.test_images, self.test_x, self.test_y, self.test_mask, dimension, clasDict, nClasses, nSteps, debug)

    def read_file(self, path, file_name, images, xs, ys, masks, dimension, clasDict, nClasses, nSteps, debug):
        with open(path + file_name) as f:
            content = f.readlines()
            for line in content:
                parts = line.split(",")
                f = parts[0]
                x = parts[1]
                img = cv2.resize(cv2.imread(path + f, cv2.IMREAD_UNCHANGED), (dimension, dimension))
                images.append(img)
                l = list(x)
                l.pop(-1)
                c = 0
                for ii in l:
                    if ii == "*":
                        break
                    c += 1
                c += 1
                masks.append([1] * c + [0] * (nSteps - c))
                l = [clasDict[i] for i in l]
                xs.append(l)
                seq = list(x)
                seq.pop(0)
                seq.pop(-1)
                y = []
                for yy in seq:
                    onehot = [0] * nClasses
                    idx = clasDict[yy]
                    onehot[idx] = 1
                    y.append(onehot)
                ys.append(y)
                print f
                if debug and len(xs) > 20:
                    break

    def next_train_batch(self, batch_size):
        startIdx = self.train_idx
        endIdx = self.train_idx + batch_size

        if endIdx >= len(self.train_x):
            endIdx = len(self.train_x) - 1

        images = self.train_images[startIdx:endIdx]
        x = self.train_x[startIdx:endIdx]
        y = self.train_y[startIdx:endIdx]
        mask = self.train_mask[startIdx:endIdx]

        self.train_idx = endIdx
        if self.train_idx >= len(self.train_x) - 1:
            self.train_idx = 0
        return x,y,images,mask

    def next_test_batch(self, batch_size):
        startIdx = self.test_idx
        endIdx = self.test_idx + batch_size

        if endIdx >= len(self.train_x):
            endIdx = len(self.train_x) - 1

        images = self.test_images[startIdx:endIdx]
        x = self.test_x[startIdx:endIdx]
        y = self.test_y[startIdx:endIdx]
        mask = self.test_mask[startIdx:endIdx]

        self.test_idx = endIdx
        if self.test_idx >= len(self.test_x) - 1:
            self.test_idx = 0
        return x,y,images,mask