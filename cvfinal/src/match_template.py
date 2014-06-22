import cv2

if __name__ == '__main__':
    personId = 6
    imgToMatch = cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + 'p.tif',0)
    template = cv2.imread('../data/Templates/01u.png',0)
    img = cv2.matchTemplate(imgToMatch, template, cv2.cv.CV_TM_SQDIFF_NORMED)
    print cv2.minMaxLoc(img) 
    cv2.imshow('img', img)
    cv2.waitKey(0)