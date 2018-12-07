from array import array
import numpy as np
from itertools import combinations

S = []
n = 6 # number of squares
k = 5 # number of pawns
w = 2 # number of white pawns
progDiv = 5 # show progress at progDiv intervals

M = list(combinations(range(n), k)) # where to put k pawns
P = list(combinations(range(k),w)) # where to put w white pawns

print("Length of M: ", len(M))

i = 0
for m in M:
    # show progess
    if i%int(len(M)/progDiv) == 0:
        print("  at: ", i/len(M))
    i += 1
    #create own pawn state
    for p in P:
        pw = np.zeros(n)
        pb = np.zeros(n)
        # kw = np.zeros(n)
        # kb = np.zeros(n)
        for j in range(k):
            if j in p:
                pw[j] = 1
            else:
                pb[j] = 1
        # s = [pw, pb, kw, kb]
        s = [pw, pb]
        S.append(s)


output_file = open('file', 'wb')
float_array = array('b', [[3, 2, 0, 1, 1]])
float_array.tofile(output_file)
output_file.close()


# 64 choose 5
# 5 choose 3
# insert kings


