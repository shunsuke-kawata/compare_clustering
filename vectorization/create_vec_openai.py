import base64
import json
import csv
from openai import OpenAI
import os
import glob
import time
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

current_time = datetime.now().strftime("%d:%H:%M")

#テストデータを保存するパス
TEST_DATA_FOLDER = './test_data_from_pss/'
OUTPUT_JSON_FILE = './caption_embedding_vector.json'
csv_output_file = f'./image_caption_data_{current_time}.csv'

client = OpenAI()

# Google Drive APIの設定
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "delta-cosmos-379819-f9ba44035375.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

#画像をBase64形式に変換する関数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#google Driveにファイルをアップロードする関数
def upload_image_to_google_drive(image_path):
    try:
        file_metadata = {'name': os.path.basename(image_path)}
        media = MediaFileUpload(image_path, mimetype='image/jpeg')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')
        
        # パブリックアクセスを設定
        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        return True,f"https://drive.google.com/uc?id={file_id}"
    except Exception as e:
        print(e)
        return False,""
        
#GPTを用いてキャプションを生成する関数
def generate_caption(encoded_image):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":"text",
                            "text":"First, explain the names of the main objects in the image in English. Next, provide a detailed description of the use or purpose of the main object. If there are multiple objects in the image, identify the main object by its location and size and describe it. The English format is “The main object is ... ,this purpose is ...” should look something like this."
                        },
                        {    
                            "type": "image_url",
                            "image_url":
                            {
                                #画像のURL
                                "url": f"data:image/jpg;base64,{encoded_image}"
                            } 
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        caption = response.choices[0].message.content
        return True, caption
    except Exception as e:
        print(e)
        return False, "Unclear"

# 画像の特徴量ベクトルを抽出する関数
def extract_feature_vector(resnet_model,image_path):

    # 特徴量ベクトルを取得するため、最終の分類層を除去
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    # 画像の前処理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 画像の読み込みと前処理の適用
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # バッチサイズの追加
    
    # 推論
    with torch.no_grad():
        feature_vector = resnet_model(input_batch)

    # ベクトルを1次元に変換
    feature_vector = feature_vector.flatten().cpu().numpy()
    return feature_vector

def extract_setence_vector(bert_model,caption):
    # Sentence-BERTモデルのロード
    bert_model = SentenceTransformer('all-MiniLM-L12-v2')
    sentence_vector = bert_model.encode(caption)  # キャプションをSentenceTransformerでベクトル化
    return sentence_vector

def create_caction_vec_in_folder(folder_path):
    
    # ResNet50モデルの読み込み（事前学習済みモデル）
    resnet_model = resnet50(pretrained=True)
    resnet_model.eval()  # 推論モードに設定
    
    # GPUが利用可能な場合はGPUを使用
    if torch.cuda.is_available():
        resnet_model = resnet_model.to('cuda')
        input_batch = input_batch.to('cuda')

    # Sentence-BERTモデルのロード
    bert_model = SentenceTransformer('all-MiniLM-L12-v2')
    
    file_list =  glob.glob(os.path.join(folder_path, '*'))
    
    with open(csv_output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Relative Path', 'File Name', 'Caption'])  # CSVヘッダー
    
        all_dict_data_list = []
        for index,file_path in enumerate(file_list):
            filename = os.path.basename(file_path) 
            print(f"start:{filename}")
            tmp_dict_data = {
                "index":index,
                "path":file_path,
                "filepath":filename
            }
            
            image_base64 = encode_image(file_path)
            
            #画像から4oを用いてキャプションを生成する
            is_success_generate_caption,generated_caption = generate_caption(image_base64)     
            if not (is_success_generate_caption):
                print(f"failed to generate caption of image :{file_path}")
                continue
            tmp_dict_data["caption"] = generated_caption
            
            #画像の特徴量とキャプションからベクトルを作成
            embedding_feature_vector = extract_feature_vector(resnet_model,file_path)
            embedding_sentence_vector = extract_setence_vector(bert_model,generated_caption)
            tmp_dict_data["embedding_feature_vector"] = embedding_feature_vector.tolist()
            tmp_dict_data["embedding_sentence_vector"] = embedding_sentence_vector.tolist()
            
            print(f"{index} finished!!!! caption and vector created: {generated_caption} \n")
            
            all_dict_data_list.append(tmp_dict_data)
            writer.writerow([index, file_path, filename, generated_caption])
            #一つ処理するごとに3秒間待機する
            time.sleep(1)
        
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(all_dict_data_list, f, indent=2)

def main():    
    create_caction_vec_in_folder(TEST_DATA_FOLDER)
    
if __name__ =='__main__':
    main()