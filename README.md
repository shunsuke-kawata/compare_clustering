# ベクトル化によるクラスタリングと画像の類似度によるクラスタリングを比較実験する

## 画像切り出しの実行

```docker compose run --rm ul```

## ChatGPT４で画像のキャプションを生成

1. 10枚ずつアップロードしてキャプションを生成する

```
For the main object in each image, write a detailed caption in English that describes what it is, its shape, color, and other characteristics, and the common use of the object. If the object is too ambiguous to create a caption, please write in the caption that it is an object that is difficult to identify.

Summarize the results in the following JSON format and list the 10 entries in the format {}, {}, {}, ....
{
    "path": "",
    "filename": "",
    "caption": "",
    "object_name": ""
},
The file extension is jpg.
For the "path" element, use the value ./results/ followed by the image file name.
```

1. caption.jsonに作成されたキャプションを保存する

## caption.jsonから文章ベクトルを生成してクラスタリングを実行する
