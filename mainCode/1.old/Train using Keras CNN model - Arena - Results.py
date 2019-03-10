#!/usr/bin/env python
# coding: utf-8

# # Experiments
#     - Test if small batch is better
#         - Exp 1
#             - Save to 006
#             - starting from 005, and shring batch size to 512
#             - Results: after 183 epochs, the accuracy was actually lower than before training. 
#               Batch size is probably not to big at 4096.
#         - Exp 2
#             - Starting from 012, a 100.000 pm net trained for 1600 epochs at bs 4096
#             - Switching to bs 32 created horrible results, network totally failed to perform, lost acc from 
#               0.95 to 0.85 in 17 epochs. Plus, super slow to train
#                  
#     - Test Adam
#         - Results
#             - 007 Adam, final accuracy 0.939 after 500
#             - 005 Adadelta, final accuracy 0.952 after 1000 epochs
#             - Adam not performing better than Adadelta
#             
#     - Test 2x2 and 3x3 filters for performance
#         - 2x2 5 layers (005)
#         - 3x3 3 layers (003)
#         - ~50.000 parameters
#         - Results
#             - 005 vs 008
#             - Same performance after 331 epochs
#             
#     - Test float16
#         - 005 float32
#         - 010 float16
#         - Results
#             - float16 scored lower
#             - float16 is 40% longer per epoch (35sec vs 50sec per epoch
#             - float32 is the way to go!
#             
#     - Try Batch Normalization
#         - model.add(BatchNormalization())
#         - Results:
#             - Comparing with 005, no batch norm
#             - Saved to 011, with batch norm
#             - ~same number of weights...
#             - Batch Normalization is performing worse after ~40 epochs
#             
#     - Try different initial weights
#         - keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
#         - Results
#             - Tried for a few epochs, similar curve as for 005
#             
#     - Test different size model
#         - Experiments
#             - 005
#                 - 50kpm
#                 - 500 epochs
#                 - 0.952 acc
#                 - still learning
#                 - 2048bs > 37s/epoch
#                 - 4096bs > 29s/epoch
#             - 013
#                 - 188kpm
#                 - 350 epochs
#                 - 0.968 acc
#                 - 150-200 epochs > stalled
#                 - 0.968 after 350 epochs
#                 - 6 hours
#             - 014
#                 - 388kpm
#                 - epochs
#                 - 0.974 after 243 epochs
#             - xxx not saved
#                 - 500kpm
#                 - 143 epochs
#                 - 0.972
#                 - slightly worse performance than 014, possibly slower to converge, but taking to long to train
#             - 016
#                 - 10kpm
#                 - 500 epochs
#                 - 0.91 acc
#             - 01
#                 - 
#         - Results
#             - 014 is best, 005 probably good enough
#             - 017 is best for fast training, maybe?
#             
#     - Measure kpm and batch size effect on calc speed
#         - not much difference
#             - #016:0.913  10kpm 2048:30                  8192:23:38% 32768:17:52% 
#             - #005:0.952  50kpm 2048:37s    4096:28:50% 
#                 - 0.94 after 100epoch = 3000s ~ 1 hour 
#             - #013:0.968 188kpm 2048:50s    4096:40s:61%             32768:46s:80% 65536:42s:99% 
#                 - 0.96 after 50 epoch = 41m
#             - #014:0.974 388kpm 2048:3m:91%
#                 - 0.97 after 60 epoch =180m
#             
#     - Test effect of bigger batch size
#         - Faster calculations, but much worse performance
#             - see 017 and 018
#             
#     - Try smaller batch size
#         - source 014
#             - 2048 batch size
#         - dest
#             - 256 bs
#         -Results
#             - started to forget, 
#             
#      - Test if increased training after validation lággildi is doing any good
#          - 022 = 388 kpm
#          - Laggildi at 70 epochs, accuracy 0.970
#          - Trained up to 250 epochs, accuracy 0.972
#          - Results
#              - it's no use to train after the 70 epoch mark
#      
#     - count wdl histogram
#     
#     - Do transfer learning
#          - Decide whats the best net
#             - 188kpm from #013, stalled at 60 epoch 4pc. 41m for 60 epochs.
#                 - 2x32-2x64-2x128-2x160-2x256
#          - Train on 3pc
#              - 024:0.998
#          - Transfer to 4pc
#              - 029: 0.964
#          - Train 4pc from scratch
#              - 028: 0.966
#          
#          
#          - How to transfer only n layers?
#          - How to freeze layers?
#          - Do n-transfer with and without freeze
#              - Use average of m runs (add average later...)
#      
#     - Find best CNN layer number
#         - trained for 250 ep
#             - 3 layers
#                 - 030
#                 - final 0.922 
#                 - time per epoch
#             - 4 layers
#                 - 032
#                 - final 0.951
#                 - time per epoch
#             - 5 layers
#                 - 028
#                 - final 0.966
#                 - time per epoch
#             - 6 layers
#                 - 034
#                 - 16 32 64 128 128 128
#                 - final 0.975
#                 - time per epoch
#             - 6 layers
#                 - 035
#                 - 16 64 64 96 128 128
#                 - final 0.975
#             - 7 layers
#                 - 056
#                 - 35kpm
#                 - 2048 batch size
#                 - final 0.956
#                 - time per epoch
#             - 7 layers
#                 - 054
#                 - 70kpm
#                 - 2048 batch size
#                 - final 0.975
#                 - time per epoch 40s
#                 - process 65%
#             - 7 layers
#                 - 050 
#                 - 184kpm
#                 - 2048 batch size
#                 - final 0.977
#                 - time per epoch 60
#                 - process 70%
#             - 7 layers
#                 - 055
#                 - 256kpm
#                 - 2048 batch size
#                 - final 0.983
#                 - time per epoch 65s
#                 - process 75%
#         - measure time to train on these nets, and decide on structure based on that
#             - bs 256 vs 2048
#                 - accuracy 256@17 = accuracy 2048@41
#                 - 256:120spb vs 2048:45spb
#                 - 2048 seems to get higher final accuracy than 256, 
#                 
#         - test different batch size for 70kpm 7 CNN layers net
#             - Results
#                 - 079  256 0.973
#                 - 080  512 0.97
#                 - 081 1024 0.973
#                 - 082 2048 0.963
#                 - 083 4096 0.967
#                 - 084 8194 0.937
#              
#         
#         
#           
#                 
#      
#      
#      
#      
#     ---------- TODO NOW --------------------------------------------------     
#     
#     
#     
#     
#     
#     - performance of 4pc seems to be negatively affected by transfer from 3pc
#         - test if I'm overfitting on 3pc, by transfering after n epochs of 3pc training
#             with n = 10, 20, 30, ...
#     
#     
#     
# 
# 
#     ---------- TODO NEXT --------------------------------------------------
#     
#      - Test if increased training after validation lággildi is doing any good
#          - Same test as above, but with smaller net (188 kpm which is MUCH faster to train)
#              - Train both with early stopping and without, and see where it stops
#                  - keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
#                    patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
# 
#             
# 
# 
# 
# 
#     ---------- TODO SOMETIME --------------------------------------------------
#     
#     - Do tree fold testing on 014, and focus on lággildi in loss at 50epochs
#     
#     - Transfer learning
#         - Read article again
#         - Make function for measurements
#             - Copy on n first layers
#             - freeze layers
#             - Average
#            
# 
# 
# 
#     - Test Checkpoints feature
#     
#     - Three fold splitting
#         - model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)
#         - https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
#         
#     - cross validation
#         - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#         
#     - Train longer....
# 
#     - Try opther optimizers
# 
#     - Make histogram of WDL values
#     
#     - Do TL experiment
#         - Take into account data split effect on TL (split 4pc at x, and then transfer to 5pc)
#         - Three split data, training, validation and testing
#         - Results
#         
#     - Finish 5pc dataset

# # WDL score count
# #### 3ps only legal
# WDLhistogram {-2=0, -1=0, 0=38368, 1=0, 2=124960}
# 
# p = [0.0, 0.235, 0.765]
# 
# #### 4pc only legal
# WDLhistogram {-2=1737970, -1=0, 0=2485090, 1=0, 2=3213028}
# 
# p = [0.234, 0.334, 0.432]
# 
# #### 5pc only legal
# WDLhistogram {-2: 20565590, -1: 0, 0: 16700358, 1: 3668, 2: 123665832}
# -2 : 20565590
# -1 : 0
# 0 : 16700358
# 1 : 3668
# 2 : 123665832
# 
# p = [0.128, 0.000, 0.104, 0.00002, 0.768]
# 
# p = [0.128,  0.104, 0.768]

# In[10]:


x3 = {-2:0, -1:0, 0:38368, 1:0, 2:124960}

x4 =  {-2:1737970, -1:0, 0:2485090, 1:0, 2:3213028}

x5 =  {-2: 20565590, -1: 0, 0: 16700358, 1: 3668, 2: 123665832}

x = x3
print(x[-2] + x[-1],x[0], x[2] + x[1] )
x = x4
print(x[-2] + x[-1],x[0], x[2] + x[1] )
x = x5
print(x[-2] + x[-1],x[0], x[2] + x[1] )


# # Number of states
# ## 4 pc
# Only pawn states = 7,436,088
# All piece states = 125,246,598	
# ## 5 pc
# Only pawn states = 160,935,448
# All piece states = 25,912,594,054 
# 

# In[6]:


x5 =20565590 + 16700358 + 3668 + 123665832
x4 =1737970 + 2485090 + 3213028
x7full = 423836835667331
x4full = 125246590
x5full = 25912594054
print('4pc',x4)
print('x4full/x4', x4full//x4)
print()
print('5pc',x5)
print('x5full/x5', x5full//x5)
print()
print('x5/x4',x5//x4)
print('x5full/x4full',x5full//x4full)
print('x7full/x4full',x7full//x4full)

trainTime4Pawns = 4
traininTime7 = x7full//x4full * x4full//x4 * trainTime4Pawns
print(traininTime7//24//365)


# # Randomly guess right probabilites
# 
# Purpose: When training net $N$ on $n$ piece dataset and using it to guess labels for a $m$ piece dataset, what is the expected accuracy of $N$ on the $m$ piece dataset? 
# 
# Methood: Given the approximation that net $N$ outputs labels randomly with probabilities $p_n$, and we sample from the $m$ piece dataset with label probabilites $p_m$, then the probability of guessing right is given by
# 
# $$ P_{n\rightarrow m} =  \sum_{i=-1}^1 p_n(i) * p_m(i)$$
# 
# Expected random sampling probability (calcRandomPerfProbability.py) for n = 3 and m = 4 we have $P_{3\rightarrow 4} = 0.409$
# 
# THIS ABOVE IS PROBABLY NO USABLE!!!!
# 
# ### !!!!!!!!!!! DO THIS AGAIN WITH THE WHOLE DATASET... !!!!!!!!!!!!!!!!!!
# 
# #### What is the expected guess-right probability given random guessing of labels
# 
# **Problem:**
# Given a random distribution A and random variable X~A, X$\in${1,2,3}, with f.x. probabilities $P_{\textrm{A}}(X = 1) = P_{\textrm{A}}(X = 2) = 0.1$ and $P_{\textrm{A}}(X = 3) = 0.8$. Let's say I'm sampling from this distribution, and also throwing a fair (non-biased) 3-sided dice with numbers 1, 2 and 3. What is the probability of the number on the dice and the sampling from the distribution agree?
# 
# **My solution**
# From intuition I would say that the probability is:
# 
# $P = \sum_{i=1}^3 P_{\textrm{Dice}}(i) * P_{\textrm{A}}(X = i)$
# 
# **The problem**
# 
# This gives us $P =  \sum_{i=1}^3 \frac{1}{3} * P_{A}(X = i) = \frac{1}{3} \sum_{i=1}^3  P_{A}(X = i) = \frac{1}{3} 1 = \frac{1}{3}$
# 
# So the distribution we are sampling from doesn't matter..., the answer is always 1/3.  I guess that might be ok, but I'm not so sure. Am I on the right track? :)
# 
# #### Results:
# **Random guessing would give us 1/3. The random net gives us 0.266 and the 3-piece net gives us 0.571. So the untrained net is worse than random, and the 3-piece net is 71\% better than random. Not great but usable... to some degree.**

# # Expansion
# 
# ### $ 3 \rightarrow 4$
# 
# RND results: $P_{RND \rightarrow 4} = 0.266$
# 
# Expanded results: $P_{3\rightarrow 4} = 0.571$
# 
# ### $ 4 \rightarrow 5$
# 
# RND results: $P_{RND \rightarrow 5} = 0.183$
# 
# Expanded results: $P_{4\rightarrow 5} = 0.523$ 

# # Training times
# 
# 
# |ds|kpm|epochs | time| tpe| t/150ep|
# |--|--|--|--|--|--|
# |4pc|70|100| 4:42 | 2.82 | 7:20|
# |4pc|181|150|7:11|2.9|7:11|

# # Results from transfer learning 
# ## $3\rightarrow 4$
# Averaged over 5 trainings, 150 epochs, results 106-115, 5 outputs
# 
# Acc$(\phi_{rnd}(D_4)) = 0.978 \pm 0.003$
# 
# Acc$(\phi_{3}(D_4)) = 0.9750 \pm 0.0003$ 
# 
# ### Results
# From https://www.socscistatistics.com/tests/studentttest/Default2.aspx: The t-value is -1.31848. The p-value is .223839. The result is not significant at p < .01.
# 
# Transfer from 3 doesn't make any difference.

# $3 \rightarrow 4$

# In[ ]:




