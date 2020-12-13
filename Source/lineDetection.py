import cv2
import numpy as np
import os
import math
from ransac_1d import *
#---------------------------------------------------- PARAMETERS ------------------------------------------------------------

kernel_size = 3 # Gaussian Kerner Size
low_threshold = 70 # Canny Threshold
high_threshold = 210 

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=7): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)

    return lines

def weighted_img(img, initial_img, α=1, β=1., λ=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

def preprocessing(image):
    height, width = image.shape[:2] # 이미지 높이, 너비
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0) # Blur 효과
    canny_img = cv2.Canny(blur_img, low_threshold, high_threshold) # Canny edge 알고리즘
    vertices = np.array([[(50,height/2+150),(width/2-45, height/2-140), (width/2+45, height/2-140), (width-50,height/2+150)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices) # ROI 설정
    ROI_img[int(height/2)-100:int(height/2)+100,int(width/2)-100:int(width/2)+100]=0
    return ROI_img

def interpolate(x1, y1, x2, y2):
    if abs(y2-y1) <10:
        return None
    
    if x2-x1 is 0:
        tilt =10000
    else:
        tilt = (y2-y1)/(x2 - x1)
    re=[]
    print("TILT = {} / {} {} {} {}".format(tilt,x1,y1,x2,y2))
    for i in range(x1, x2, 3):
        idx = i-x1
        re.append([x1+idx, y1+tilt*idx])
        # print(" {} {}".format([x1+idx, y1+tilt*idx],idx))
    return re
#----------------------------------------------------READ IMAGE FILE------------------------------------------------------------
path = os.getcwd()
subPath = '\hg_lineDetection\Data'
fileName = '\slope_test.jpg'
fileName = '\cart2.png'

totalPath = path+subPath+fileName
# print(totalPath+"asdfasdf")
image = cv2.imread(totalPath)
#----------------------------------------------------READ IMAGE FILE------------------------------------------------------------

# cv2.imshow('temp',image)
canny_img = preprocessing(image)


line_arr = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환
line_arr = np.squeeze(line_arr)
# print(line_arr)
# print(line_arr.shape)
# 기울기 구하기
slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi
# print(slope_degree.shape)


# 수평 기울기 제한
# line_arr = line_arr[np.abs(slope_degree)<160]
# slope_degree = slope_degree[np.abs(slope_degree)<160]
# # 수직 기울기 제한
# line_arr = line_arr[np.abs(slope_degree)>95]
# slope_degree = slope_degree[np.abs(slope_degree)>95]
# # 필터링된 직선 버리기
# L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
# temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
# L_lines, R_lines = L_lines[:,None], R_lines[:,None]

temp_L = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
temp_R = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

L_lines, R_lines = line_arr[:], line_arr[:]
L_lines, R_lines = L_lines[:,None], R_lines[:,None]



L_Line=[]
R_Line=[]

L_Point=[]
R_Point=[]
Lx=[]
Rx=[]
Ly=[]
Ry=[]
for i in line_arr:
    # print(i)
    reArray=interpolate(i[0],i[1],i[2],i[3])
    if reArray is None:
        print("NONE")
    elif i[1] < i[3]:
        R_Line.append(i)
        R_Point.append(reArray)
        for re in reArray:
            Rx.append(re[0])
            Ry.append(re[1])
    else:
        L_Line.append(i)
        L_Point.append(reArray)
        for re in reArray:
            Lx.append(re[0])
            Ly.append(re[1])

# print(L_Point)
# print(np.array(L_Line)[:,None])
draw_lines(temp_L, np.array(L_Line)[:,None])
draw_lines(temp_R, np.array(R_Line)[:,None])

# 직선 그리기
# draw_lines(temp, np.array(L_Line[:,None]))
# draw_lines(temp, np.array(R_Line[:,None]))
# draw_lines(temp, R_lines)
# print(line_arr)
# print(line_arr[:,None])
# print(np.squeeze(np.squeeze(np.squeeze(L_Point))))
result_L = weighted_img(temp_L, image) # 원본 이미지에 검출된 선 overlap
result_R = weighted_img(temp_R, image) # 원본 이미지에 검출된 선 overlap
cv2.imshow('resultL',result_L) # 결과 이미지 출력
cv2.imshow('resultR',result_R) # 결과 이미지 출력
print("AAAAAAAAa")
print(Lx)
print(Ly)

model_a, model_b, model_c = RANSAC(Lx,Ly)
print("MODEL {} {} {}".format(model_a, model_b, model_c))
model_Lx = [a for a in range(100, 300)]
model_Ly = [model_a*a+int(model_c) for a in range(100, 300)]


plt.plot(Lx,Ly,'*b')
plt.plot(model_Lx,model_Ly,'*r')
plt.show()

# print("MODEL {} {} {}".format(model_a, model_b, model_c))




cv2.waitKey(0)