'''
Created on 13-mei-2014

@author: vital.dhaveloose
'''

import matplotlib.pyplot as ppl

'''
Takes a matrix of the form Landmarks x Tooth x Dimension and shows it on the screen.
'''
def plotTeeth(matrix):
    nbT = matrix.shape[1]
    
    for tooth in range(nbT-1):
        plotTooth(matrix[:,tooth+1,:])

'''
Takes a matrix of the form Landmark x Dimension and shows it on the screen
'''
def plotTooth(matrix):
    ppl.plot(matrix[:,0], matrix[:,1])    

'''
Takes a matrix of the form Landmark x Dimension and shows it on the screen
'''
def plotToothXY(x,y):
    ppl.plot(x,y)

def show():
    ppl.axis('equal')
    ppl.gca().invert_yaxis()
    ppl.show()