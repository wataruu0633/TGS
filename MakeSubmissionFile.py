import csv
import numpy as np
from PIL import Image
def rle_encoding(x):
    dots = np.where(x.T.flatten()==65535)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

sample = open('./data/train.csv')
reader = csv.DictReader(sample)
f = open('submission.csv', 'w')

f.write('id,rle_mask\n')

for row in reader:
    img = np.array(Image.open('./data/train/masks/' + row['id'] + '.png'))
    rle = rle_encoding(img)
    rle_st = ' '.join(map(str,rle))
    f.write(row['id'] + ',' + rle_st + "\n")
