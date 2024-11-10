import json
from create_vector import create_json_with_vector

def main():
    # JSONファイルのパス
    json_file_path = 'caption.json'

    # ファイルからJSONデータを読み込む
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    json_data_with_vector = create_json_with_vector(json_data=json_data)
    
    with open("caption_embedding_sentence_vector.json", "w") as f:
        json.dump(json_data_with_vector, f, indent=2)

if __name__ =='__main__':
    main()

