import requests
import json
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import base64
import urllib


def get_access_token_qianfan():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=DTNX9fIESYPHsOLYRSgGEPBJ&client_secret=PUfU7ew9kXJOBpZRzlmkY3OBRZ7ksB1m"
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_zidongtaichu( question ):
    # 紫东太初文本生成
    api = 'https://ai-maas.wair.ac.cn/maas/v1/chat/completions'
    headers = {'Authorization': 'Bearer tlf3tc8sltk89etyx0p16u5p'}
    params = {
        'model': 'taichu_llm',
        'messages': [{"role": "user", "content": f"{question}"}],
        'stream': False
    }

    response = requests.post(api, json=params, headers=headers, stream=True)
    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        return content
    else:
        body = response.content.decode('utf-8')
        print(f'request failed,status_code:{response.status_code},body:{body}')


def get_qianfan_text( question ):
    # 千帆文本生成
    access_token = get_access_token_qianfan()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + str(access_token)
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"{question}"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data = payload )
    if response.status_code == 200:
        response_json = response.json()
        content = response_json[ 'result' ]
        return content
    else:
        body = response.content.decode('utf-8')
        print(f'request failed,status_code:{response.status_code},body:{body}')


def get_qianfan_graph( question ):
    access_token = get_access_token_qianfan()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/sd_xl?access_token=" + str( access_token )
    payload = json.dumps({
        "prompt": f"{question}",
        "size": "1024x1024",
        "n": 1,
        "steps": 20,
        "sampler_index": "Euler a"
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200:
        body = response.content.decode('utf-8')
        print(f'request failed,status_code:{response.status_code},body:{body}')
    response_dict = response.json()
    b64_image = response_dict['data'][0]['b64_image']

    # plt显示图片
    image_data = base64.b64decode( b64_image )
    image = Image.open(BytesIO(image_data))
    plt.imshow( image )
    plt.axis('off')
    plt.show()


def get_qianfan_read( question ):  # 这里只能传入图片的地址
    access_token = get_access_token_qianfan()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b?access_token=" + access_token
    question = image_to_base64( question )

    payload = json.dumps( {
        "prompt": "解释一下这张图片",
        "image": f"{question}"
    } )
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request( "POST", url, headers = headers, data = payload )
    response_dict = response.json()
    return response_dict[ 'result' ]


def image_to_base64(image_path):
    with open( image_path, "rb" ) as image_file:
        image_data = image_file.read()
        base64_encoded = base64.b64encode( image_data ).decode( 'utf-8' )
        return base64_encoded


def api_check(n, question):

    if n == '1':
        result = get_qianfan_text( question )
    elif n == '2':
        result = get_zidongtaichu( question )
    elif n == '3':
        result = get_qianfan_graph( question )
    elif n == '4':
        result = get_qianfan_read( question )
    print( result )

