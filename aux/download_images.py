from test_ids_trie import ManageTrie
from datetime import datetime
import urllib
import os

with open('train.txt') as f:
        count = 0

        trie = ManageTrie()

        for line in f:
                count += 1
                img1, img2, lbl = line.rstrip().split(' ')
                img1 = img1.split('/')[-1]
                img2 = img2.split('/')[-1]

                if not trie.lookup_in_trie(img1):
                        trie.add_to_trie(img1)

                        if not os.path.isdir("clothingstyle/images/I/%s/%s/%s" % (img1[0], img1[1], img1[2])):
                                os.makedirs("clothingstyle/images/I/%s/%s/%s" % (img1[0], img1[1], img1[2]))

                        with open("clothingstyle/images/I/%s/%s/%s/%s" % (img1[0], img1[1], img1[2], img1), 'w') as f:
                                f.write(urllib.urlopen("http://ecx.images-amazon.com/images/I/"+img1).read())

                if not trie.lookup_in_trie(img2):
                        trie.add_to_trie(img2)

                        if not os.path.isdir("clothingstyle/images/I/%s/%s/%s" % (img2[0], img2[1], img2[2])):
                                os.makedirs("clothingstyle/images/I/%s/%s/%s" % (img2[0], img2[1], img2[2]))

                        with open("clothingstyle/images/I/%s/%s/%s/%s" % (img2[0], img2[1], img2[2], img2), 'w') as f:
                                f.write(urllib.urlopen("http://ecx.images-amazon.com/images/I/"+img2).read())

                #if count%1000==0:
                #        print count/1999999.0, datetime.time(datetime.now())
