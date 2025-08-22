# from models.cnn_model import CNNModel
from models.inception_model import InceptionModel

from config import Config
from PIL import Image, ImageDraw, ImageFont
import torch
from utils.data_loader import get_test_transforms
from pathlib import Path


def process_images(input_dir="need_predict"):
    """处理need_predict文件夹中的所有图片"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 获取所有支持的图片格式
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in input_dir.glob('*') if p.suffix.lower() in image_exts]

    if not image_paths:
        print(f"警告: {input_dir} 中没有找到可处理的图片")
        return []

    return image_paths


def add_prediction_banner(original_path, prediction, confidence):
    """在原始图片上方添加白色横幅区域显示预测结果"""
    output_dir = Path("predicted")
    output_dir.mkdir(exist_ok=True)

    original_img = Image.open(original_path).convert('RGB')
    original_width, original_height = original_img.size

    banner_height = 80
    new_img = Image.new(
        'RGB',
        (original_width, original_height + banner_height),
        color=(255, 255, 255)
    )
    new_img.paste(original_img, (0, banner_height))

    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    text = f"{prediction} ({confidence})"
    text_width = draw.textlength(text, font=font)
    text_position = ((original_width - text_width) // 2, 20)

    draw.text(text_position, text, fill=(0, 0, 0), font=font)

    new_path = output_dir / f"{original_path.stem}_with_result{original_path.suffix}"
    new_img.save(new_path)
    return str(new_path)


def predict_single_image(model, image_path, device):
    """预测单张图片"""
    transform = get_test_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prob = torch.nn.functional.softmax(output, dim=1)[0] * 100

    classes = ['cat', 'dog']
    return {
        'class': classes[predicted.item()],
        'confidence': f"{prob[predicted.item()]:.2f}%"
    }


def main():

    # 初始化模型
    # model = CNNModel(Config.num_classes).to(Config.device)
    model = InceptionModel(Config.num_classes).to(Config.device)

    model.load_state_dict(torch.load(Config.model_path, weights_only=True))

    # 处理所有待预测图片
    image_paths = process_images()

    for img_path in image_paths:
        print(f"\n处理图片: {img_path.name}")
        try:
            result = predict_single_image(model, img_path, Config.device)
            new_path = add_prediction_banner(img_path, result['class'], result['confidence'])
            print(f"预测结果: {result['class']} (置信度: {result['confidence']})")
            print(f"结果保存到: {new_path}")
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")


if __name__ == '__main__':
    main()