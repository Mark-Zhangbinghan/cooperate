import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import requests
import argparse

# 自建使用函数
import MAS_Function             # MAS收敛函数
import model_judge as judge     # 判断模型类型并输出结果
import Model_api as api         # 调用大模型的函数
import download                 # 下载模型函数

"""
##  符号解释
""" """         三个双引号代表其中的内容可以直接替换
#               井号表示注释

##  变量含义
in_type         判断输入的是图片或文字  文字：0  图片：1
value           串行中用来接收和返回模型的输入和输出
value_table     并行过程中用来接收所有模型的输出
file_path       读取的json文件的路径
model_type      模型的类型（是mindir还是onnx）
img_path        图片地址
weight_matrix   权重矩阵

##  函数含义
read_json(file_path)            根据json文件来进行串行或并行操作
get_value(model_path)           分析模型后缀获取模型类型并调用模型
"""


AK = "F915WYG9INM7JCMKWYA8"
SK = "bWMR0xxcVBxOA6URk86efzREAOXLzoZvu6lkU00M"
ENDPOINT = "https://obs.cn-south-1.myhuaweicloud.com"

# 指定桶名称、文件键（OBS中文件的路径和文件名）、本地文件夹
bucket_name = "qg23onnx"
object_key = "onnx/resnet50-CLDC.onnx"  # OBS中文件的完整路径
local_folder = "download"  # 你希望保存下载文件的本地文件夹


# 这里设置读取模型返回的值
def read_json(data):
    # 设置大模型对应的字典
    switcher = {
        'get_qianfan_text': '1',
        'get_zidongtaichu': '2',
        'get_qianfan_graph': '3',
        'get_qianfan_read': '4'
    }

    try:
        if not data["content"]:
            value = data["image"]
            in_type = 1  # 输入是图片
            """这里要确定图片是给的地址，如有需要在这里添加读取图片的代码"""
        else:
            value = data["content"]  # 文字直接读取即可
            in_type = 0  # 输入是文字
        for layer_data in data["modelList"]:
            # 判断串并行条件
            if layer_data["parallel"] == 0:
                print("串行")
                for model in layer_data["models"]:  # 开始串行
                    print(model["isAPI"])
                    # 判断模型是本地还是调用大模型
                    if model["isAPI"] == 0:  # 模型为本地
                        model_path = download.download_file(bucket_name, object_key, local_folder, ENDPOINT, AK, SK)
                        # 在函数内部判断模型是mindir还是onnx
                        value = judge.get_value(model["modelName"], model_path, value, in_type)

                    else:   # 模型调用大模型
                        n = switcher.get(model["modelName"])
                        value = api.api_check(n, value)

            else:
                print("并行")                # 目前只支持文字输出
                value_tabel = []            # 用来接收所有输出
                weight_matrix = []          # 设置权重矩阵
                for model in layer_data["models"]:
                    weight_matrix.append(model["weight"])

                    # 判断模型是本地还是调用大模型
                    if model["isAPI"] == 0:  # 模型为本地
                        model_path = download.download_file(bucket_name, object_key, local_folder, ENDPOINT, AK, SK)
                        value_tabel.append(judge.get_value(model["modelName"], model_path, value, in_type))

                    else:   # 模型调用大模型
                        n = switcher.get(model["modelName"])
                        value_tabel.append(api.api_check(n, value))

                value = MAS_Function.change_value(value_tabel, weight_matrix)
                print(value)
    except Exception as e:
        print("错误类型：", type(e).__name__)
        print("错误信息：", str(e))

    data.update({"answer": str(value)})

