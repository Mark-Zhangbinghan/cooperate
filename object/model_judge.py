import sys  # 用于程序直接退出
import general_mindir as gm     # mindir模型读取通用代码（目前只支持读入图片）
import image_pred_onnx as image_pred
#import emotion_analyse_onnx as emotion_analyse
#hghfgjh
# 用来获取模型的类型
def get_model_type(model_path):
    model_types = {
        'onnx': 0,
        'mindir': 1
    }
    extension = model_path.split('.')[-1]
    return model_types.get(extension, -1)  # 如果后缀名不在字典中，返回-1


def get_value(model_name, model_path, value, in_type):
    model_type = get_model_type(model_path)

    if model_type == 0:  # 模型是onnx文件
        if in_type == 1:
            value = image_pred.use_model(model_path, value)
#        else:
#            value = emotion_analyse.main(vocab_path, model_path, value)

    elif model_type == 1:  # 模型是mindir文件
        if in_type == 1:
            value = gm.main_mindir(model_path, value)

    else:  # 模型类似不符
        print("请使用mindir或onnx类型的模型")
        sys.exit(0)  # 直接退出程序

    return value
