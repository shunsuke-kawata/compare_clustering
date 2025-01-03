import json
from create_vector import create_json_with_vector


def create_vec_from_json():
    # JSONファイルのパス
    json_file_path = 'caption.json'

    # ファイルからJSONデータを読み込む
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    json_data_with_vector = create_json_with_vector(json_data=json_data)
    
    with open("caption_embedding_vector.json", "w") as f:
        json.dump(json_data_with_vector, f, indent=2)
    print("created result json")
    

def main():
    return


if __name__ =='__main__':
    main()

