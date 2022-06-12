'''
Description: 
Author: xianpengfei
LastEditors: xianpengfei
Date: 2022-05-26 21:02:23
LastEditTime: 2022-06-11 14:04:07
'''
from paddleocr import PaddleOCR, draw_ocr
import os, base64
from flask import Flask, request, jsonify
from flask_restful import Api,Resource
import jieba
import jieba.analyse as analyse
import json
import numpy as np
import base64



# 图片识别
def distinguish_img(img_path):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch",enable_mkldnn=True, use_gpu=False, use_tensorrt= True)  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    res = []
    for item in result:
        res.append({
            "coords": item[0],
            "data": {
                "text":item[1][0],
                "value": item[1][1]
            }
        })
    return res
    os.remove(img_path)

# base64转图片
def base64_to_img(bstr, file_path):
    imgdata = base64.b64decode(bstr)
    file = open(file_path, 'wb')
    file.write(imgdata)
    file.close()
    print('successful')

app = Flask(__name__)
api = Api(app)

@app.route('/')
def hello_world():
    return 'AI - 识别服务'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class ImageView(Resource):
    def get(self):
        return {'code':200,'msg':'plese use post method!!!'}
    def post(self):
        file_path = './image/test.png'
        images = request.json.get('images')
        datas = []
        for item in images:
            base64_to_img(item, file_path)
            res = distinguish_img(file_path)
            datas.append(res)
        print(np.array(datas).tolist())
        return {'code': 200, 'msg': 'ok', 'data': str(datas)}

class JieBaView(Resource):
  def post(self):
    form = request.json
    print(form)

    topK = form.get('topK', 100)
    body = form.get('body', 'hello')
    excludes = form.get('excludes', [])

    cut_all = bool(form.get('is_cut_all', False))

    data = analyse.extract_tags(body, topK=topK, withWeight=True)

    newList = []

    for x in data:
        if x[0] in excludes:
            print('not')
        else:
            newList.append({
                "data": x[0],
                "value": x[1]
            })

    print(newList)

    return {
        'code': 200,
        'msg': 'ok',
        'datas': newList,
    }


api.add_resource(ImageView, '/image-to-text')
api.add_resource(JieBaView, '/jieba-text')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8866)
