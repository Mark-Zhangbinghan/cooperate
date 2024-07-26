import os
from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore import context
import mindspore as ms
import numpy as np
from PIL import Image
import mindspore.dataset.vision.transforms as transforms
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE)
# 定义 argmax 操作
argmax = ops.Argmax(axis=1)


def preprocess_image(image_path):
    """预处理输入图像以进行预测。"""
    trans = [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ]
    img = Image.open(image_path).convert('RGB')
    for transform in trans:
        img = transform(img)
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    return Tensor(img, mstype.float32)


def load_model(model_path):
    """加载模型并返回模型实例。"""
    try:
        graph = ms.load(model_path)
        model = nn.GraphCell(graph)
        model.set_train(False)
        return model
    except Exception as e:
        # 打印错误信息并重新抛出异常
        print(f"Error loading model: {e}")
        raise


def get_tensor_shape(model, image_path):
    """获取模型输出的张量形状。"""
    image = preprocess_image(image_path)
    output = model(image)
    return output.shape


def predict(model, image_path):
    labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                      8: 'ship', 9: 'truck'}
    """使用给定模型预测输入图像的标签。"""
    image = preprocess_image(image_path)
    output = model(image)
    label = labels[argmax(output).asnumpy()[0]]
    return label  # 返回模型的输出


# mindir模型使用的通用
def main_mindir(model_path, img_path):
    model = load_model(model_path)

    if os.path.isfile(img_path):
        output = predict(model, img_path)
    return output
