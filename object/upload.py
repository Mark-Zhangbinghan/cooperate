import os
from obs import ObsClient


def upload_file(bucket_name, object_key, file_path, endpoint, ak, sk):
    """
    上传指定的OBS对象到本地文件夹。

    :param bucket_name: OBS桶名称
    :param object_key: OBS中对象的键
    :param file_path: 本地文件夹路径，上传文件的本地路径
    :param endpoint: OBS服务终端节点
    :param ak: 访问密钥ID
    :param sk: 密钥访问密钥
    """
    # 创建ObsClient实例
    obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=endpoint)

    # 构造本地文件的完整路径

    try:
        # 执行下载操作
        obsClient.putFile(bucket_name, object_key, file_path=file_path)
        url = endpoint + "/" + object_key
        print(f"successful upload:{url}")
        return url
    except Exception as e:
        print(f"error：{e}")
