with open("testBig.txt") as infile:
    j = 0
    for line in infile:
        i = line
        print(j)
        j += 1
