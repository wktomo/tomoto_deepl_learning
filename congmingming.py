import os
import re

def rename_files_in_folder(folder_path):
    # 获取文件夹中所有文件和文件夹
    entries = os.listdir(folder_path)
    # 过滤出文件（排除文件夹）
    files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
    # 按文件名排序（逆序，从大到小）
    files.sort(reverse=True)
    
    # 定义替换规则，去除空格和特殊字符
    def sanitize_filename(filename):
        # 使用正则表达式替换非字母、数字、下划线的字符
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
        # 去除文件名中的空格
        sanitized = sanitized.replace(' ', '_')
        return sanitized
    
    # 重命名文件
    for index, filename in enumerate(files, start=1):
        # 构造新的文件名
        new_filename = f"{index}_{sanitize_filename(filename)}"
        # 构造完整的文件路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")

# 示例用法
folder_path = 'D:\\suanfa\\tomato-disease\\val_data'  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)