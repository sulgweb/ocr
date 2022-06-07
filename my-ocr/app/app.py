'''
Description: 
Author: xianpengfei
LastEditors: xianpengfei
Date: 2022-05-26 21:02:23
LastEditTime: 2022-06-08 00:41:02
'''
from paddleocr import PaddleOCR, draw_ocr
import os, base64
from flask import Flask, request
from flask_restful import Api,Resource
import paddle
from paddlespeech.cli.asr import ASRExecutor
from paddlespeech.cli.text import TextExecutor
from paddlespeech.server.bin.paddlespeech_client import ASROnlineClientExecutor


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
    return 'AI - 识别服务'

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
        return {'code': 200, 'msg': 'ok', 'data': str(datas)}

class AudioView(Resource):
  def get(self):
    device = paddle.device.get_device()
    asr_executor = ASRExecutor()
    text = asr_executor(
        model='conformer_wenetspeech',
        lang='zh',
        sample_rate=16000,
        config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
        ckpt_path=None,
        audio_file='./audio/001.wav',
        force_yes=True,
        device=device)
        
    print('ASR Result: \n{}'.format(text))
    return {'code': 200, 'msg': 'ok', 'data': str(text)}


class PunctuationView(Resource):
  def get(self):
    text_executor = TextExecutor()
    result = text_executor(
        text='今天的天气真不错啊你下午有空吗我想约你一起去吃饭',
        task='punc',
        model='ernie_linear_p3_wudao',
        lang='zh',
        config=None,
        ckpt_path=None,
        punc_vocab=None,
        device=paddle.get_device())
    print('Text Result: \n{}'.format(result))
    return {'code': 200, 'msg': 'ok', 'data': str(result)}


api.add_resource(ImageView, '/image-to-text')
api.add_resource(AudioView, '/audio-to-text')
api.add_resource(PunctuationView, '/replay-punctuation')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8866)
