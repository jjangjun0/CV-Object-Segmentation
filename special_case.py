import cv2
import numpy as np
import pydicom
import os
import sys
from pydicom.uid import generate_uid   
from pydicom.uid import generate_uid

''' global variable '''
clicked_point_y = None

# 마우스 콜백 함수
def click_event(event, x, y, flags, param):
    global clicked_point_y
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 눌렀을 때
        print(f"클릭 좌표: x={x}, y={y}")
        clicked_point_y = x
        
        img_result = param.copy()
        cv2.circle(img_result, (x, y), radius=5, color=255, thickness=-1)
        cv2.imshow("Click pixel in image to guess what is not included object's area", img_result)

def open_dcm(file: str):
    # 0-1. DICOM 이미지 읽기
    dcm = pydicom.dcmread(file)
    # 0-2. 명암대비 스트레칭
    pixel_min = np.min(dcm.pixel_array)
    pixel_max = np.max(dcm.pixel_array)
    img = (dcm.pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
    # 0-3. float64 -> uint8 로 변환
    ct_img = img.astype(np.uint8)
    # cv2.imshow(file, ct_img)
    return ct_img

# file을 읽고 객체 경계를 파악한다.
def making_output(file: str, ret1: int):
    global clicked_point_y
    ct_img = open_dcm(file)
    # CLANE 알고리즘 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(ct_img)
    # ret 결정 => a * ret1(ret_avg) + b * ret2(현재 이미지의 Otsu thresholding으로 구한 임계값)
    ret2, otsu_binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    split = int((0.2)*ret1 + (0.8)*ret2)
    print(f"ret1: {ret1}, ret2: {ret2}, split: {split}")

    # 마우스 콜백 함수 등록
    cv2.imshow("Click pixel in image to guess what is not included object's area", enhanced_img)
    cv2.setMouseCallback("Click pixel in image to guess what is not included object's area", click_event, enhanced_img)
    # 사용자 클릭 대기
    
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    # 클릭 값 확인
    if clicked_point_y is None:
        print("! 클릭이 발생하지 않았습니다 !")
        sys.exit(0)

    rows2, cols2 = enhanced_img.shape
    #fill_pixel = enhanced_img[0][int(cols2/2)]
    fill_pixel = enhanced_img[0][0]
    ### print(fill_pixel)

    for j in range(0, clicked_point_y):
        for i in range(rows2):
            enhanced_img[i][j] = fill_pixel


    # 1. 이진화 -> Otsu Thresholding
    ret3, binary_img = cv2.threshold(enhanced_img, split, 255, cv2.THRESH_BINARY)

    # 2. 열기 연산 실행 (침식 -> 팽창)
    kernel = np.ones((9, 9), np.uint8)
    iterations = 1
    erode_img = cv2.erode(binary_img, kernel, iterations)
    dilate_img = cv2.dilate(erode_img, kernel, iterations)

    binary_img = dilate_img
    ### cv2.imshow("binary_img", binary_img)

    # 3. 윤곽선 검출(객체 경계 파악)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 외곽선 찾기
    largest_contour = max(contours, key=cv2.contourArea)
    contours_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_img, [largest_contour], -1, (255, 0, 0), 2)
    ### cv2.imshow("Contours", contours_img)
    ### cv2.waitKey(0)

    # 4. 마스크 영역 설정
    mask_img = np.zeros_like(binary_img)
    rows, cols = mask_img.shape
    cv2.drawContours(mask_img, [largest_contour], -1, (255), thickness=cv2.FILLED)
    # 4-1 열에 대해서 마스크를 채운다.
    # numpy의 경우, 행과 열이 바뀌어 저장되어 있다.
    for j in range(cols):
        start = 0
        end = 0
        for i in range(rows):
            if (mask_img[i][j] != 0):
                start = i
                break
        for i in range(rows-1, -1, -1):
            if (mask_img[i][j] != 0):
                end = i
                break

        # print(f"start: {start}, end: {end}")
        for i in range(start, end+1):
            mask_img[i][j] = 255
    ### cv2.imshow("mask_img2", mask_img)
    ### cv2.waitKey(0)
    # 4-2 CT 이미지라면 분명 가운데에 객체가 검출되도록 간호사가 안내한다.
    # 즉, 가운데에 검출되지 않았다면 다시 행에 대해서 마스크를 채운다.
    # SPECIAL_CASE의 경우 검사 범위를 10 -> 30 으로 증가
    is_middle_pixel = 0
    temp_x = 30
    temp_y = 30
    half_x = int(cols / 2)
    half_y = int(rows / 2)
    for j in range(half_x - temp_x, half_x + temp_x):
        for i in range(half_y - temp_y, half_y + temp_y):
            if (mask_img[i][j] == 0):
                is_middle_pixel = 1
                break

    if (is_middle_pixel):
        for i in range(rows):
            start = 0
            end = 0
            for j in range(cols):
                if (mask_img[i][j] != 0):
                    start = j
                    break
            for j in range(cols-1, -1, -1):
                if (mask_img[i][j] != 0):
                    end = j
                    break
        
            # print(f"start: {start}, end: {end}")
            for j in range(start, end+1):
                mask_img[i][j] = 255
    ### cv2.imshow("mask_img3", mask_img)
    ### cv2.waitKey(0)

    # 5. 마스크 영역 색깔 설정
    overlay = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    mask = cv2.bitwise_not(mask_img)
    overlay[mask == 0] = (0, 0, 255)  # BGR => Red color(255)

    overlay_alpha = 0.3               # 투명도

    # 6. 이미지와 overlay 이미지를 겹친다.
    img_color = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(overlay, overlay_alpha, img_color, 1 - overlay_alpha, 0, img_color)
    
    ### cv2.imshow("mask_img", mask_img)
    ### cv2.imshow("overlay", overlay)
    ### cv2.imshow("img_color", img_color)

    # mask_img => Output
    # img_color => Overlay
    return mask_img, img_color
def saving_file(file: str, i: int, dcm_dir, png_dir, mask_img, img_color):
    # 결과 이미지 이름 설정
    idx_str = "{:03}".format(i)
    Output = "Output_" + idx_str
    Overlay = "Overlay_" + idx_str

    dcm = pydicom.dcmread(file)

    # 1. 마스크 DICOM 이미지 저장
    mask_dcm = dcm.copy()
    mask_dcm.SamplesPerPixel = 1
    mask_dcm.PhotometricInterpretation = "MONOCHROME2"
    mask_dcm.BitsAllocated = 8
    mask_dcm.BitsStored = 8
    mask_dcm.HighBit = 7
    mask_dcm.PixelRepresentation = 0
    mask_dcm.PixelData = mask_img.tobytes()
    mask_dcm.Rows, mask_dcm.Columns = mask_img.shape
    mask_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    mask_dcm.SOPInstanceUID = generate_uid()
    mask_dcm.save_as(dcm_dir + "/" + Output + ".dcm")

    # BGR을 RGB로 변환 => DICOM 파일은 RGB로 저장되어야 한다.
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # 2. Overlay DICOM 이미지 저장
    overlay_dcm = dcm.copy()
    overlay_dcm.PhotometricInterpretation = "RGB"
    overlay_dcm.Rows, overlay_dcm.Columns, overlay_dcm.SamplesPerPixel = img_color_rgb.shape
    overlay_dcm.BitsAllocated = 8
    overlay_dcm.BitsStored = 8
    overlay_dcm.HighBit = 7
    overlay_dcm.PixelRepresentation = 0
    overlay_dcm.PixelData = img_color_rgb.tobytes()
    overlay_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    overlay_dcm.SOPInstanceUID = generate_uid()
    overlay_dcm.save_as(dcm_dir + "/" + Overlay + ".dcm")

    # 3. PNG 파일로 저장
    cv2.imwrite(png_dir + "/" + Output + ".png", mask_img)
    cv2.imwrite(png_dir + "/" + Overlay + ".png", img_color)

'''
# 개별 실행 #
mask_img, img_color = making_output('Case 4/dcm/Case4_373.dcm', 90)
cv2.imshow("mask_img", mask_img)
cv2.imshow("img_color", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# 메인 실행
li = []
li.append('Case 4/dcm/Case4_366.dcm')
li.append('Case 4/dcm/Case4_367.dcm')
li.append('Case 4/dcm/Case4_368.dcm')
li.append('Case 4/dcm/Case4_369.dcm')
li.append('Case 4/dcm/Case4_370.dcm')
li.append('Case 4/dcm/Case4_371.dcm')
li.append('Case 4/dcm/Case4_372.dcm')
li.append('Case 4/dcm/Case4_373.dcm')
li.append('Case 4/dcm/Case4_374.dcm')
li.append('Case 4/dcm/Case4_375.dcm')
li.append('Case 4/dcm/Case4_376.dcm')
li.append('Case 4/dcm/Case4_377.dcm')

# 디렉토리 생성
result_dir_path = "Case 4 Result"
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

dcm_dir = result_dir_path + "/dcm"
if not os.path.exists(dcm_dir):
    os.makedirs(dcm_dir)

png_dir = result_dir_path + "/png"
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

# 속도 개선을 위해 dcm_dir의 평균 임계값은 미리 구한다.
RET_AVG = 90

nums = []
for path in li:
    filename = path.split('/')[-1]       # 'Case4_370.dcm'
    number = filename.replace('.dcm', '')[-3:]  # 마지막 3자리
    nums.append(number)

print(nums)

for i in range(len(li)):
    mask_img, img_color = making_output(li[i], RET_AVG)
    cv2.imshow("mask_img", mask_img)
    cv2.imshow("img_color", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    saving_file(li[i], nums[i], dcm_dir, png_dir, mask_img, img_color)