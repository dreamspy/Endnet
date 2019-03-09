# script for taking in directory lists of results directories and 
# extracting accuracy values and save to pickled dict

#command used to create data
# ls -1 -R | grep -E 'tablebase|__8' |grep -v ./ |grep -v experimentDesc

import os
import pickle

def save_obj(saveDir, saveName, obj ):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    fileName = saveDir + '/'+ saveName + '.pkl'
    # askAbortIfPathExists(fileName)
    with open(fileName, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, fileName ):
    with open(dir + '/' + fileName + '.pkl', 'rb') as f:
        return pickle.load(f)

def isDigit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

d3 = {'drawAccuracy':[],
'drawCount':[],
'finalAccuracy':[],
'finalCount':[],
'lossAccuracy':[],
'lossCount':[],
'winAccuracy':[],
'winCount':[]}

d4 = {'drawAccuracy':[],
'drawCount':[],
'finalAccuracy':[],
'finalCount':[],
'lossAccuracy':[],
'lossCount':[],
'winAccuracy':[],
'winCount':[]}

f = open("3to4.txt", "r")
switch = False
	
for l in f:
	if '3PKk' in l:
		switch = False
	if '4PpKk' in l:
		switch = True
	if "__8" in l:
		start = l.find('__8.') + 4
		end = l.find('---')
		digits = [s for s in l.split() if isDigit(s)]
		key = l[start:end]
		value = digits[0]
		if switch:
			d4[key].append(value)
		else:
			d3[key].append(value)

print("3.........")
for key, value in d3.items():
	print(key, value)
print("4.........")
for key, value in d4.items():
	print(key, value)


save_obj('.','d3',d3)
save_obj('.','d4',d4)

# s = 'gfgfdAAA1234ZZZuijjk'
# start = s.find('AAA') + 3
# end = s.find('ZZZ', start)



# print(f.readline())
# {'finalAccuracy'':[val1,val2],
#  'finalCount'':[val1,val2],
#  ...
#  }



