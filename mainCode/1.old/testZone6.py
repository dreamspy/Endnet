s = [i for i in range(64)]
S=[]
r = 1000000000
for i in range(r):
    S.append(s)
    if i%(r/1000)==0:
        print(100*i/r, i)
print(len(S))
input("press any key to terminate")