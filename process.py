from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os

imageName = list(filter(lambda file: file[-3:] == 'jpg', os.listdir()))
columnDir = 'splitColumn'
resultsDir = 'results'
for image in imageName:
    print(image)
    img = cv2.imread(image)
    (H, W) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(3,3),0)
    ret, thes = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Split page
    pages = [thes[:, 0:round(W/2)], thes[:, round(W/2):W]]
    colList = []

    # Crop header & footer (New method but not complete)
    # for pageNum, page in enumerate(pages):
    #     cv2.imwrite("test.jpg", page)
    #     edges = cv2.Canny(page, 150, 200)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 300, maxLineGap=250)
    #
    #     threshold_lines = []
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         if (x1 < 50) or (x1 > page.shape[1] - 50):
    #             continue
    #         threshold_lines.append(line)
    #
    #     line = threshold_lines[0]
    #     # print(line)
    #
    #     contours, _= cv2.findContours(edges, 1, 2)
    #     thresholdContours = []
    #     for contour in contours:
    #         approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.04, True)
    #         arcLength = cv2.arcLength(contour, True)
    #         if arcLength < 300 or arcLength > 1000:
    #             continue
    #         thresholdContours.append(contour)
    #
    #
    #     contourCloestToBottom = thresholdContours[0]
    #     print(contourCloestToBottom)
    #     M = cv2.moments(contourCloestToBottom)
    #     currentMaxY = int(M['m01']/M['m00'])
    #
    #     for i in range(0, len(thresholdContours)):
    #         M = cv2.moments(thresholdContours[i])
    #         contourY = int(M['m01']/M['m00'])
    #
    #         if currentMaxY < contourY:
    #             currentMaxY = contourY
    #             contourCloestToBottom = thresholdContours[i]
    #
    #     cv2.drawContours(page, [contourCloestToBottom], -1, (0,255,0), 10)
    #
    #     currentMaxY +=  200 #padding bottom
    #     line[0][1] += 20 #padding top
    #
    #     # pages[pageNum] = page[line[0][1]:(currentMaxY-line[0][1]), 0: page.shape[1]]
    #     cv2.imshow("image", cv2.resize(page, (640, 640)))
    #     cv2.waitKey(0)


    # Skew correction for each column
    for i, page in enumerate(pages):
        gray = cv2.bitwise_not(page)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), ang = ret
        if w>h:
            w,h = h,w
            ang += 90

        M = cv2.getRotationMatrix2D((cx,cy), ang, 1.06)
        rotated = cv2.warpAffine(page, M, (page.shape[1], page.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pages[i] = rotated
        # cv2.imshow('image', cv2.resize(pages[i], (480,640)))
        # cv2.waitKey(0)

    # Crop header & footer (Old method)
    for i, col in enumerate(pages):
        th = 250
        h, w = col.shape[:2]

        dst = cv2.erode(col, kernel=np.ones((30, 30)))
        hist = cv2.reduce(dst,1, cv2.REDUCE_AVG).reshape(-1)

        uppers = [y for y in range(h-1) if hist[y]>th and hist[y+1]<=th]
        upper = max(list(filter(lambda x: x < H / 2, uppers)))

        dst = cv2.erode(col, kernel=np.ones((15, 30)))
        hist = cv2.reduce(dst,1, cv2.REDUCE_AVG).reshape(-1)
        th = 235
        lowers = [y for y in range(h-1) if hist[y]>th and hist[y+1]<=th]
        # print(lowers)
        lower = min(list(filter(lambda x: x > 7 * H / 8, lowers)))
        #
        expand = 5
        # for upper in lowers:
        #     cv2.line(col, (0,upper - expand), (w, upper - expand), (0,255,0), 3)
        # cv2.line(col, (0,lowers + expand), (w, lowers + expand), (0,255,0), 3)

        pages[i] = (col[upper - expand:lower + expand, 0:w])
        # cv2.imshow('image', cv2.resize(col[upper - expand:lower + expand, 0:w], (480,640)))
        # cv2.waitKey(0)

    # Skew correction for each column
    for i, col in enumerate(pages):
        gray = cv2.bitwise_not(col)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), angle = ret
        if angle < -45:
        	angle = -(90 + angle)
        else:
        	angle = -angle

        M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
        rotated = cv2.warpAffine(col, M, (col.shape[1], col.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pages[i] = rotated
        # cv2.imshow('image', cv2.resize(rotated, (480,640)))
        # cv2.waitKey(0)



    # Crop column
    for i, page in enumerate(pages):
        # cv2.imshow('image', cv2.resize(page, (480, 640)))
        # cv2.waitKey(0)
        h, w = page.shape[:2]

        dst = cv2.erode(page, kernel=np.ones((30, 10)))
        hist = cv2.reduce(dst, 0, cv2.REDUCE_AVG).reshape(-1)
        th = min(list(filter(lambda x: x > 245 and x < 255, hist)))
        uppers = [y for y in range(w-1) if hist[y]>th and hist[y+1]<=th]

        if (i == 0):
            # dst = cv2.erode(page, kernel=np.ones((60, 10)))
            # hist = cv2.reduce(page, 0, cv2.REDUCE_AVG).reshape(-1)
            th = min(list(filter(lambda x: x > 250 and x < 255, hist)))
        else:
            hist = cv2.reduce(page, 0, cv2.REDUCE_AVG).reshape(-1)
            th = min(list(filter(lambda x: x > 240 and x < 255, hist)))
        lowers = [y for y in range(w-1) if hist[y]<=th and hist[y+1]>th]
        # print(uppers)
        # print(lowers)
        # for k in range(len(uppers)):
        #     cv2.line(page, (uppers[k], 0), (uppers[k], h), (0,255,0), 3)
        # for k in range(len(lowers)):
        #     cv2.line(page, (lowers[k], 0), (lowers[k], h), (0,255,0), 3)
        #
        # cv2.imshow('image', cv2.resize(page, (480, 640)))
        # cv2.waitKey(0)

        potentialPair = []
        for iU in range(len(uppers)):
            for iL in range(len(lowers)):
                # print([uppers[iU]])
                if (uppers[iU] < lowers[iL] and abs(uppers[iU] - lowers[iL]) > w / 4 and abs(uppers[iU] - lowers[iL]) < w / 2):
                    potentialPair.append([iU, iL])
        # print(potentialPair)
        leftCols = list(filter(lambda x: uppers[x[0]] < w / 3, potentialPair))
        rightCols = list(filter(lambda x: lowers[x[1]] > 2 * w / 3, potentialPair))
        # print(len(rightCols))

        if (len(leftCols) == 0 or len(rightCols) == 0):
            print("File " + image + "error! Please recheck it.")
        leftCol = min(leftCols, key=lambda x: lowers[x[1]] - uppers[x[0]])
        rightCol = min(rightCols, key=lambda x: lowers[x[1]] - uppers[x[0]])

        expand = round((lowers[leftCol[1]] - uppers[leftCol[0]]) * 0.03)
        colList.append(page[0:h,uppers[leftCol[0]] - expand:lowers[leftCol[1]] + expand])

        expand = round((lowers[rightCol[1]] - uppers[rightCol[0]]) * 0.03)
        colList.append(page[0:h,uppers[rightCol[0]] - expand:lowers[rightCol[1]] + expand])


    # Skew correction for each column
    for i, col in enumerate(colList):
        gray = cv2.bitwise_not(col)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), angle = ret
        if angle < -45:
        	angle = -(90 + angle)
        else:
        	angle = -angle

        M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
        rotated = cv2.warpAffine(col, M, (col.shape[1], col.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        colList[i] = rotated
        cv2.imwrite(columnDir + '/' + image[0:-4] + '-' + str(i) + '.jpg', rotated)

    # Split line
    lineCrop = []
    colIndex = 0
    for eachCol in colList:
        dst = cv2.erode(eachCol, kernel=np.ones((1, 30)))
        hist = cv2.reduce(dst,1, cv2.REDUCE_AVG).reshape(-1)

        th = 230
        h, w = eachCol.shape[:2]
        uppers = [y for y in range(h-1) if hist[y]>th and hist[y+1]<=th]
        uppers.append(h)

        for i in range(len(uppers) - 1):
            expand = 5
            if (uppers[i+1] - uppers[i] < 15):
                continue
            crop = eachCol[uppers[i] - expand:uppers[i+1], 0:w]
            lineCrop.append(crop)

            # cv2.line(eachCol, (0,uppers[i] - expand), (w, uppers[i+1] - expand), (0,255,0), 3)

            # cv2.imshow('image', crop)
            # cv2.waitKey(0)

    # Skew correction for each line
    for i, line in enumerate(lineCrop):
        gray = cv2.bitwise_not(line)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), ang = ret
        ang += 90
        if w>h:
            w,h = h,w
            ang -= 90

        M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
        rotated = cv2.warpAffine(line, M, (line.shape[1], line.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        lineCrop[i] = rotated

        # cv2.imshow('image', line)
        # cv2.waitKey(0)


    config = ("-l vie --oem 1 --psm 7")
    outputText = []
    for crop in lineCrop:
        outputText.append(pytesseract.image_to_string(crop, config=config))
        # print(pytesseract.image_to_string(crop, config=config))
        # outputText.append(pytesseract.image_to_data(crop, config=config))
        # a = pytesseract.image_to_pdf_or_hocr(crop, extension='hocr', config=config)
        # with open("result.xml", "wb") as wf:
        #     wf.write(a)

    # for text in outputText:
    #     print(text)
    with open(resultsDir + '/' + image[0:-3] + 'txt', 'w', encoding='utf8') as f:
        for text in outputText:
            f.write(text + '\n')
