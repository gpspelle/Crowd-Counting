'''
Crowd-Counting using Multi-Column Convolutional Neural Networks.
'''

from network import MCNN
#import sys

'''
# For accepting the input from the prompt.
if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('Usage: python3 test.py A(or B)')
    exit()
'''

dataset='B'

'''
# For the B dataset.
dataset = 'B'
'''

#path = 'black/'
path = 'green/'
mcnn = MCNN(dataset)

# For predicting the count of people in one Image.
#images = ['DSCN1636.JPG', 'DSCN1637.JPG', 'DSCN1638.JPG', 'DSCN1639.JPG', 'DSCN1640.JPG', 'DSCN1641.JPG']
mcnn.predict(path)

'''
# For predicting the count of people in all Images in the Test Dataset.
mcnn.test()
'''
