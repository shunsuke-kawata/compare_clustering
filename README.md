# ベクトル化によるクラスタリングと画像の類似度によるクラスタリングを比較実験する

## 画像切り出しの実行

```docker compose run --rm ul```

## ChatGPT４で画像のキャプションを生成

1. 10枚ずつアップロードしてキャプションを生成する

```
For each image, create a detailed caption in English describing what the main object might be, how it is used, its shape, color, and other features. For unclear objects, include a tentative guess with an indication that the identification may be uncertain.

Summarize the results in the following JSON format, listing 10 entries in the form {},{},{},...
{
    "path": "",
    "filename": "",
    "caption": "",
    "object_name": ""
},
For the "path" element, use the value ./result/ followed by the image file name.
```

1. caption.jsonに作成されたキャプションを保存する

## caption.jsonから文章ベクトルを生成してクラスタリングを実行する
