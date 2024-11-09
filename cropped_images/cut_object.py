import os
import cv2
import uuid
import glob
import pyheif
from PIL import Image
from ultralytics import YOLO
import shutil

MODEL_PATH = "yolov10x.pt"

def convert_heic_to_jpeg(heic_file_path, jpeg_file_path):
    heif_file = pyheif.read(heic_file_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(jpeg_file_path, "JPEG")

def process_images(image_url_list, tmp_data_dir):
    for image_url in image_url_list:
        # 画像の拡張子を取得
        _, ext = os.path.splitext(image_url)

        # JPEG形式の画像はそのままtmp_dataフォルダにコピー
        if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
            shutil.copy(image_url, tmp_data_dir)
        else:
            # HEICなどJPEG以外の画像はJPEGに変換
            jpeg_file_path = os.path.join(tmp_data_dir, f'{uuid.uuid1()}.jpg')
            convert_heic_to_jpeg(image_url, jpeg_file_path)

def main():
    print("start")
    model = YOLO(MODEL_PATH)
    
    class_names = model.names
    for class_name in class_names.items():
        print(class_name[0],class_name[1])
        
    print("loaded model")
    
    save_dir = './results'
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    draw_dir = './results_with_boxes'
    shutil.rmtree(draw_dir)
    os.makedirs(draw_dir, exist_ok=True)
    
    tmp_data_dir = './tmp_data'
    os.makedirs(tmp_data_dir, exist_ok=True)

    image_url_list = glob.glob('./data/*')
    
    # 画像の処理
    process_images(image_url_list, tmp_data_dir)
    
    index = 1

    # tmp_data内の画像を使って推論の実行
    for image_url in glob.glob(os.path.join(tmp_data_dir, '*')):
        results = model.predict(image_url, save=False, conf=0.80, show=False)
        print("ended predict")

        image = cv2.imread(image_url)
        image_copy = image.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[box.cls[0].item()]
                conf = box.conf[0].item()
                cropped_image = image[y1:y2, x1:x2]

                new_path = os.path.join(save_dir, f'{index}_{label}.jpg')
                index +=1
                cv2.imwrite(new_path, cropped_image)
                print(f'Cropped file saved as: {new_path}')
                
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_copy, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        drawn_image_path = os.path.join(draw_dir, f'drawn_image_{uuid.uuid1()}.jpg')
        cv2.imwrite(drawn_image_path, image_copy)
        print(f'Drawn image saved as: {drawn_image_path}')

    # tmp_dataフォルダを削除
    shutil.rmtree(tmp_data_dir)
    
    os.remove(MODEL_PATH)
    print("deleted model")
    print("all done")

if __name__ == '__main__':
    main()
