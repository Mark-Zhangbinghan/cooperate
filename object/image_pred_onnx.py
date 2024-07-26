import os
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


def load_and_preprocess_images(image_path, shape_list):
    images = []
    image_ids = []
    all_image_tensors = []
    # 定义预处理流程，但不包括归一化
    preprocess = transforms.Compose([
        transforms.Resize((shape_list[0][2], shape_list[0][3])),  # 确保所有图像大小一致
        transforms.ToTensor()
    ])

    # 读取所有图片并进行预处理
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    images.append(input_tensor)
    filename = image_path.split('\\')[-1]
    image_ids.append(filename)  # 记录图像文件名
    all_image_tensors.append(input_tensor)

    # 将图像张量堆叠起来，转换成numpy数组进行计算
    images_np = torch.stack(all_image_tensors).numpy()
    # 计算整体的均值和标准差
    mean = np.mean(images_np, axis=(0, 2, 3))
    std = np.std(images_np, axis=(0, 2, 3))
    # 创建归一化变换
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    # 对每个图像张量进行归一化
    normalized_images = [normalize_transform(image) for image in all_image_tensors]
    return normalized_images, image_ids


def use_model(onnx_model_path, image_folder):
    ort_session = ort.InferenceSession(onnx_model_path)
    input_metadata = ort_session.get_inputs()
    shape_list = []
    # 遍历输入张量并打印shape
    for input_tensor in input_metadata:
        # 获取输入张量的名称和shape
        tensor_shape = input_tensor.shape
        shape_list.append(tensor_shape)

    input_images, image_ids = load_and_preprocess_images(image_folder, shape_list)
    # 确保 input_images 是一个批次的图像，即增加了批次维度
    if len(input_images) == 3:  # 假设是单张图像，需要增加批次维度
        input_images = np.expand_dims(input_images, axis=0)

    ort_inputs = {input_metadata[0].name: input_images}
    ort_outs = ort_session.run(None, ort_inputs)

    predicted_class = np.argmax(ort_outs[0], axis=1).item()  # 提取预测结果并转换为普通整数
    # 输出预测结果
    # 映射字典
    disease_map = {
        "0": "木薯细菌性枯萎病（CBB）",
        "1": "木薯褐条纹病（CBSD）",
        "2": "木薯绿斑驳 （CGM）",
        "3": "木薯花叶病（CMD）",
        "4": "健康"
    }
    # 将预测结果映射为疾病名称
    predicted_diseases = disease_map[str(predicted_class)]
    return predicted_diseases
