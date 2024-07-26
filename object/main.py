import json
from download import download_file
from read_json import read_json


if __name__ == "__main__":
    """
    # OBS配置信息
    AK = "F915WYG9INM7JCMKWYA8"
    SK = "bWMR0xxcVBxOA6URk86efzREAOXLzoZvu6lkU00M"
    ENDPOINT = "https://obs.cn-south-1.myhuaweicloud.com"

    # 指定桶名称、文件键（OBS中文件的路径和文件名）、本地文件夹
    bucket_name = "qg23onnx"
    object_key = "Upload/2024QG工作室人工智能组中期考核-基于MindSpore的多智能体协作系统.md"  # OBS中文件的完整路径
    local_folder = "download"  # 你希望保存下载文件的本地文件夹

    # 调用函数下载文件
    download_file(bucket_name, object_key, local_folder, ENDPOINT, AK, SK)
    """
    file_path = r"C:\Users\24468\Desktop\QG人工智能组\Python学习\实战\middle_exam\aiApi.json"
    with open(file_path, 'r', encoding='utf-8') as file:  # 这里打开模型返回的json文件，并寻找其中的关键字
        data = json.load(file)
    value = read_json(data)

