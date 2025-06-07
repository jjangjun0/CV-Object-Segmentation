import cv2
import numpy as np
import pydicom
import os
from pydicom.uid import generate_uid

''' Functions '''
def open_dcm(file: str):
    # 0-1. DICOM 이미지 읽기
    dcm = pydicom.dcmread(file)
    # 0-2. 명암대비 스트레칭
    pixel_min = np.min(dcm.pixel_array)
    pixel_max = np.max(dcm.pixel_array)
    img = (dcm.pixel_array - pixel_min) / (pixel_max - pixel_min) * 255
    # 0-3. float64 -> uint8 로 변환
    ct_img = img.astype(np.uint8)
    ### cv2.imshow("ct_img", ct_img)
    return ct_img

# dir_path에 있는 data들을 가지고, Otsu thresholding을 했을 때의 임계값들의 평균을 반환하는 함수
def find_ret_avg(li: list[str], NUM: int):
    ret_avg = 0
    for i in range(NUM+1):
        img = open_dcm(li[i])
        ret1, otsu_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret_avg += ret1
    ret_avg /= NUM
    ret_avg = int(ret_avg)
    return ret_avg

# file을 읽고 객체 경계를 파악한다.
def making_output(file: str, ret1: int):
    ct_img = open_dcm(file)

    # ret 결정 => a * ret1(ret_avg) + b * ret2(현재 이미지의 Otsu thresholding으로 구한 임계값)
    ret2, otsu_binary = cv2.threshold(ct_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    split = int((0.1)*ret1 + (0.9)*ret2)
    print(f"ret1: {ret1}, ret2: {ret2}, split: {split}")

    # 1. 이진화 -> Otsu Thresholding
    ret3, binary_img = cv2.threshold(ct_img, split, 255, cv2.THRESH_BINARY)

    # 2. 열기 연산 실행 (침식 -> 팽창)
    kernel = np.ones((11, 11), np.uint8)
    iterations = 1
    erode_img = cv2.erode(binary_img, kernel, iterations)
    dilate_img = cv2.dilate(erode_img, kernel, iterations)

    binary_img = dilate_img
    ### cv2.imshow("binary_img", binary_img)

    # 3. 윤곽선 검출(객체 경계 파악)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 외곽선 찾기
    largest_contour = max(contours, key=cv2.contourArea)
    ### contours_img = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)
    ### cv2.drawContours(contours_img, [largest_contour], -1, (255, 0, 0), 2)
    ### cv2.imshow("Contours", contours_img)
    ### cv2.waitKey(0)

    # 4. 마스크 영역 설정
    mask_img = np.zeros_like(binary_img)
    cv2.drawContours(mask_img, [largest_contour], -1, (255), thickness=cv2.FILLED)

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

# 메인 실행
''' Case 3 data '''
NUM = 150           # 데이터 개수
dir_path = "Case 3"
file_type = "dcm"
file_name = "Case3_"

li = []
for i in range(NUM+1):
    idx_str = "{:03}".format(i)
    absolute_path = f"{dir_path}/{file_type}/{file_name}{idx_str}.{file_type}"
    li.append(absolute_path)
### print(li)

RET_AVG = find_ret_avg(li, NUM) # 특정 디렉토리 하위의 data들을 가지고, 리스트를 만든 후에 평균값을 미리 구한다.

# 디렉토리 생성
result_dir_path = "Case 3 Result"
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

dcm_dir = result_dir_path + "/dcm"
if not os.path.exists(dcm_dir):
    os.makedirs(dcm_dir)

png_dir = result_dir_path + "/png"
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

for i in range(NUM+1):
    mask_img, img_color = making_output(li[i], RET_AVG)
    saving_file(li[i], i, dcm_dir, png_dir, mask_img, img_color)


print("ALL save")