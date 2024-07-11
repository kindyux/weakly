import os
from PIL import Image

# 输入文件夹路径
input_folder = 'datasets/ICPR2024/masks_centroid'
# 输出文件夹路径
output_folder = 'datasets/ICPR2024/reshaped_masks'
# 目标大小 (宽, 高)
target_size = (512, 512)
# 二值化阈值
threshold = 40

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 只处理图片文件
        try:
            # 打开图片
            img = Image.open(os.path.join(input_folder, filename))
            # 调整大小
            img = img.resize(target_size, Image.LANCZOS)
            # 转换为灰度图
            gray_img = img.convert('L')
            # 应用二值化阈值
            binarized_img = gray_img.point(lambda p: 255 if p > threshold else 0)
            # 保存到输出文件夹
            binarized_img.save(os.path.join(output_folder, filename))
            print(f"Processed {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print("All images processed.")
