import cv2
import numpy as np

import main
import radiograph as rg
import landmarks as lm
import procrustes as procru

'''
A template is a tuple (img,lms) with img is the image of the
template, and lms are the (translated) landmarks of the template.
'''

'''
Creates a template from the given personId (image) and tooth. Delta is the extra space around the landmarks in the returned image.s
'''
def getTemplate(personId, toothIds, Delta=20):
    # read landmarks and image
    landmarks = lm.readLandmarksOfPersonAndTeeth(personId, toothIds)
    image = rg.readRadioGraph(personId)
    
    # crop image to box around landmarks
    minX = np.min(landmarks[:,:,0])-Delta
    maxX = np.max(landmarks[:,:,0])+Delta
    minY = np.min(landmarks[:,:,1])-Delta
    maxY = np.max(landmarks[:,:,1])+Delta
    image = image[minY:maxY,minX:maxX]
    
    # translate landmarks to the coordinate system of the cropped image
    for i in range(toothIds.shape[0]):
        landmarks[:,i,:] = procru.translateMatrixForPerson(landmarks[:,i,:], np.array([[-minX],[-minY]]))
    
    return image, landmarks

'''
Returns a triplet which contains the location, scale resp. score of the best match of the template image in the image

angles in radians, 0 being opwards and + in counter clockwise rotation
'''
def matchTemplate(image, templImage, scales = np.array([0.8, 0.9, 1.0, 1.1, 1.2])):
    chosenScr = 0
    chosenLoc = (0,0)
    chosenScale = 0
    
    # find score, location and scale of best match
    for scale in scales:
        scaled_templ = cv2.resize(templImage.copy(), (0,0), fx=scale, fy=scale)
        # avoid scaling too large
        if scaled_templ.shape[0] < image.shape[0] and scaled_templ.shape[1] < image.shape[1] :
            matches = cv2.matchTemplate(image, scaled_templ, cv2.cv.CV_TM_SQDIFF_NORMED)
            scr, _, loc, _ = cv2.minMaxLoc(matches)
            # invert, because SQDIFF gives lower results for better matches
            if scr == 0 : return loc, scale, scr
            scr = 1.0-scr
            if scr > chosenScr:
                chosenScr = scr
                chosenLoc = loc
                chosenScale = scale
                
    if chosenScale == 0 : print 'The given template image does not fit in the given image for any scale.'
    
    return chosenLoc, chosenScale, chosenScr

def templFilter(img):
    return img
    gaussKern = 9
    gaussSigma = 7
    laplaceKern = 7
    blobMinArea = 400

    # gradients with laplace
    img = cv2.Laplacian(img, cv2.CV_32FC1, ksize = laplaceKern)/256
    # gaussian blur
    img = cv2.GaussianBlur(img,(gaussKern,gaussKern),sigmaX=gaussSigma,sigmaY=gaussSigma)
    # inversion
    img = np.ones(img.shape)*255-img
    # recoding to unsignes int, 8bit
    img = np.uint8(img)
    # thresholding
    img= cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    
    # find contours
    contours = np.array(cv2.findContours(img.copy(), cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0])
    
    for i in range(contours.shape[0]):
        # if countour of a small area
        if cv2.contourArea(contours[i]) < blobMinArea:
            # re-order coordinates (because they are a real mess)
            ordered = np.zeros((np.array(contours[i]).shape[0], 2))
            for k in range(np.array(contours[i]).shape[0]):
                ordered[k,0] = contours[i][k][0][0]
                ordered[k,1] = contours[i][k][0][1]
            # overdraw blobs with white
            cv2.fillPoly(img, np.int32([ordered]), 255)
    
    return img

def testCreateTemplate():
    for personId in range(14):
        print str(personId+1)
        image, landmarks = getTemplate(personId, range(0))
        for toothId in range(main.toothIds.shape[0]):
            cv2.polylines(image, np.int32([landmarks[:,toothId,:]]), True, 255)
        cv2.imshow('result', image)
        cv2.waitKey(0)

def testMatchTemplate():
    for personId in range(14,30):
        image = rg.readRadioGraph(personId)
        
        chosenScr = 0
        for templId in range(14):
            template = getTemplate(templId, np.array(range(8)))
            (x,y), scale, scr = matchTemplate(image, template[0])
            if scr > chosenScr:
                chosenScr = scr
                chosenX = x
                chosenY = y
                chosenTemplate = template
                chosenTemplId = templId
                
        print (personId, chosenTemplId)
        
        main.showScaled(template[0], 0.66, 'template', True)
        main.showScaled(templFilter(image), 0.66,'image', True)
        
        combined = image.copy()
        #combined[x:x+template[0].shape[0], y:y+template[0].shape[1]] = template[0]
        cv2.circle(combined, (y,x), 3, 128, thickness=-1)
        #main.showScaled(combined, 0.66, 'combined', False)
        
        init_image = image.copy()
        for toothId in range(8):
            init_points = procru.translateMatrixForPerson(chosenTemplate[1][:,toothId,:], np.array([[chosenX],[chosenY]]))
            cv2.polylines(init_image, np.int32([init_points]), True, 255)
        cv2.imshow('init_image',init_image)
        main.showScaled(init_image, 0.66, 'init_image', True)

def testMatchTemplateFilter():
    for personId in range(14,30):
        # WITHOUT FILTER
        image = rg.readRadioGraph(personId)
        
        sumScrs = 0
        avgTeeth = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
        corrLM = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
        
        for templId in range(14):
            # get the template
            templImg, templLM = getTemplate(templId, np.array(range(8)))
            # match the template
            (x,y), scale, scr = matchTemplate(image, templImg)
            # translate the template landmarks
            for i in range(main.toothIds.shape[0]):
                toothIdx = main.toothIds[i]
                corrLM[:,toothIdx,:] = procru.scaleMatrixForPerson(templLM[:,toothIdx,:], scale)
                corrLM[:,toothIdx,:] = procru.translateMatrixForPerson(corrLM[:,toothIdx,:], np.array([[x],[y]]))
            # add the translated landmarks to the average tooth with weight = scr
            avgTeeth = avgTeeth + scr**main.templ_scr_loyalty*corrLM
            # update the sum of all scores
            sumScrs = sumScrs + scr**main.templ_scr_loyalty
            
        avgTeeth = avgTeeth/sumScrs
        
        init_image = image.copy()
        for toothId in range(8):
            cv2.polylines(init_image, np.int32([avgTeeth[:,toothId,:]]), True, 255)
        main.showScaled(init_image, 0.66, 'without filter', True)
        
        # WITH FILTER
        image = rg.readRadioGraph(personId)
        
        sumScrs = 0
        avgTeeth = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
        corrLM = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
        
        for templId in range(14):
            # get the template
            templImg, templLM = getTemplate(templId, np.array(range(8)))
            # match the template
            (x,y), scale, scr = matchTemplate(templFilter(image), templFilter(templImg))
            # translate the template landmarks
            for i in range(main.toothIds.shape[0]):
                toothIdx = main.toothIds[i]
                corrLM[:,toothIdx,:] = procru.scaleMatrixForPerson(templLM[:,toothIdx,:], scale)
                corrLM[:,toothIdx,:] = procru.translateMatrixForPerson(corrLM[:,toothIdx,:], np.array([[x],[y]]))
            # add the translated landmarks to the average tooth with weight = scr
            avgTeeth = avgTeeth + scr**main.templ_scr_loyalty*corrLM
            # update the sum of all scores
            sumScrs = sumScrs + scr**main.templ_scr_loyalty
            
        avgTeeth = avgTeeth/sumScrs
        
        init_image = image.copy()
        for toothId in range(8):
            cv2.polylines(init_image, np.int32([avgTeeth[:,toothId,:]]), True, 255)
        main.showScaled(init_image, 0.66, 'with filter', True)

if __name__ == '__main__':
    
    testMatchTemplateFilter()
    '''
    for personId in range(30):
        img = rg.readRadioGraph(personId)
        main.showScaled(img, 0.66, 'before', True)
        img = templFilter(img)
        main.showScaled(img, 0.66, 'after', True)
        
    personId = 6
    imgToMatch = cv2.imread(main.radiographPath(personId, True),0)
    template = cv2.imread('../data/Templates/01u.png',0)
    img = cv2.matchTemplate(imgToMatch, template, cv2.cv.CV_TM_SQDIFF_NORMED)
    print cv2.minMaxLoc(img) 
    cv2.imshow('img', img)
    cv2.waitKey(0)
    '''
