from paddleocr import PaddleOCR, draw_ocr
import os, base64
from flask import Flask, request
from flask_restful import Api,Resource


# 图片识别
def distinguish_img(img_path):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch",enable_mkldnn=True, use_gpu=False)  # need to run only once to download and load model into memory
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
    return 'OCR 识别服务'

class DistinguishView(Resource):

    # get接口提示
    def get(self):
        return {'code':200,'msg':'plese use post method!!!'}

    # post接口
    def post(self):
        file_path = './image/test.png'
        images = request.json.get('images')
        datas = []
        for item in images:
            base64_to_img(item, file_path)
            res = distinguish_img(file_path)
            datas.append(res)
        return {'code': 200, 'msg': 'ok', 'data': str(datas)}

api.add_resource(DistinguishView, '/distinguish')
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8866)
