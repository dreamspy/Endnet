# Calculate estimated success rate when transfering network from 3pc to 4pc, withut retraining.
import numpy as np

count3 = np.array([0, 38368,124960])
total3 = count3.sum()
count4 = np.array([1737970, 2485090, 3213028])
total4 = count4.sum()

p3 = [i/total3 for i in count3]
p4 = [i/total4 for i in count4]

P = np.array([p3[i]*p4[i] for i in range(3)]).sum()
print("When training net N on 3pc dataset, and using it to estimate a 4pc dataset.Then the estimated probability of guessing right, given that the net guesses randomly for an outcome with probabilities p3 and p4, is P = ",\
      round(P,3))
