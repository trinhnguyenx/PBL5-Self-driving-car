import cv2
import numpy as np
import utils

curveList = []
avgVal = 10

def getLaneCurve(img, display=2):
    
    h, w, c = img.shape
    points = np.float32([(102, 80), (480-102, 80), (20, 214), (480-20, 214)])

    imgCopy = img.copy()
    imgResult = img.copy()
    imgThres = utils.thresholding(imgCopy)
    imgWarp = utils.warpImg(imgThres, points, w, h)
    imgWarpPoints = utils.drawPoints(imgCopy, points)
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)

    curveRaw = curveAveragePoint - middlePoint
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, w, h, True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:h//3, 0:w] = 0,0,0
        
        imgLaneColor = np.zeros_like(img)
        imgLaneColor = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)

        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)

        midY = 450
        cv2.putText(imgResult,str(curve),(w//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
        cv2.line(imgResult,(w//2,midY),(w//2+(curve*3),midY),(255,0,255),5)
        cv2.line(imgResult, ((w // 2 + (curve * 3)), midY-25), (w // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = w // 20
            cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                        (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
    if display == 2:
        imgStacked = utils.stackImages(0.9,([img,imgWarpPoints,imgWarp],
                                            [imgHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt',imgResult)

    return curve

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')

    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (480, 240))
            curve = getLaneCurve(img, 2)
            print(curve)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)