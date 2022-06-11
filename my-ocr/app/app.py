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
import numpy as np



# 图片识别
def distinguish_img(img_path):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch",enable_mkldnn=True, use_gpu=False, use_tensorrt= True)  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    return result
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

class ImageView(Resource):
    def get(self):
        return {'code':200,'msg':'plese use post method!!!'}
    def post(self):
        file_path = 'app/image/test.png'
        images = request.json.get('images')
        datas = []
        for item in images:
            base64_to_img(item, file_path)
            res = distinguish_img(file_path)
            datas.append(res)
        return {'code': 200, 'msg': 'ok', 'data': str(datas)}

class JieBaView(Resource):
  def get(self):
    test_sent = """
    数学是一门基础性的大学课程，深度学习是基于数学的，尤其是线性代数课程
    """
    words = jieba.cut(test_sent)
    data = np.array(list(words))
    print(data)
    return {'code': 200, 'msg': 'ok', 'data': 1}


api.add_resource(ImageView, '/image-to-text')
api.add_resource(JieBaView, '/jieba-text')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8866)
