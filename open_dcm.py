import cv2
import numpy as np
import pydicom

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
    cv2.imshow(file, ct_img)
    return ct_img

def open_dcm_color(file: str):
    ds = pydicom.dcmread(file)
    print(f"PhotometricInterpretation: {ds.PhotometricInterpretation}")

    if ds.PhotometricInterpretation == "RGB":
        rows, cols = ds.Rows, ds.Columns
        samples_per_pixel = ds.SamplesPerPixel
        if samples_per_pixel != 3:
            raise ValueError("RGB 이미지인데 SamplesPerPixel이 3이 아닙니다.")

        # RGB 데이터 읽고 (H, W, 3)로 reshape
        arr = np.frombuffer(ds.PixelData, dtype=np.uint8)
        rgb_image = arr.reshape(rows, cols, 3)

        # RGB → BGR (OpenCV는 BGR 사용)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB DICOM (as BGR)", bgr_image)
        return bgr_image
    else:
        print('Nothing opened')

'''확인해보고 싶은 dcm 파일'''
# case directory 설정
dir = '4'
# index_str 설정 ex) 003, 018, 333 ...
index_str = '376'
# file_type
type = '.dcm'

file = "Case "+ dir +"/dcm/Case"+ dir +"_"+ index_str + type
open_dcm(file)
file = "Case "+ dir +" Result/dcm/Overlay_"+ index_str + type
open_dcm_color(file)
file = "Case "+ dir +" Result/dcm/Output_" + index_str + type
open_dcm(file)

cv2.waitKey(0)
cv2.destroyAllWindows()