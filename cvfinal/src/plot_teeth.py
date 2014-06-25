'''
Created on 13-mei-2014

@author: vital.dhaveloose
'''

import matplotlib.pyplot as ppl

'''
Takes a matrix of the form Point x Tooth x Dimension and shows it on the screen.
'''
def plotTeeth(matrix):
    nbT = matrix.shape[1]
    
    for toothId in range(nbT):
        plotTooth(matrix[:,toothId,:])

'''
Takes a matrix of the form Landmark x Dimension and shows it on the screen
'''
def plotTooth(matrix):
    ppl.plot(matrix[:,0], matrix[:,1])    

'''
Takes a matrix of the form Landmark x Dimension and shows it on the screen
'''
def plotToothXY(xStacked,y):
    ppl.plot(xStacked,y)

def show():
    ppl.autoscale()
    ppl.axis('equal')
    ppl.gca().invert_yaxis()
    ppl.show()