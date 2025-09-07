#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此脚本用于修复Markdown文件中的图片引用路径问题。
它会将Markdown文件中的图片引用从相对路径转换为基于工作空间根目录的绝对路径，
确保在VSCode预览中能正确显示图片。
"""

import os
import re
import argparse
import shutil
from datetime import datetime

# 支持的图片文件扩展名
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp']


def is_image_file(filename):
    """检查文件是否为图片文件"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def get_all_markdown_files(root_dir):
    """获取所有Markdown文件的路径"""
    markdown_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.md'):
                full_path = os.path.join(dirpath, filename)
                markdown_files.append(full_path)
    return markdown_files


def fix_markdown_images(md_file_path, workspace_root):
    """修复Markdown文件中的图片引用路径"""
    # 创建备份文件
    backup_path = f"{md_file_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(md_file_path, backup_path)
    
    # 读取Markdown文件内容
    with open(md_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 获取Markdown文件所在的目录
    md_dir = os.path.dirname(md_file_path)
    
    # 匹配Markdown中的图片引用: ![alt](path)
    img_pattern = r'!\[(.*?)\]\((.*?)\)'
    matches = re.findall(img_pattern, content)
    
    if not matches:
        print(f"{md_file_path}: 未找到图片引用")
        return False
    
    changes_made = False
    
    for alt_text, img_path in matches:
        # 跳过已经是绝对路径的图片引用
        if img_path.startswith(('http://', 'https://', 'file:///')):
            continue
        
        # 跳过空路径
        if not img_path.strip():
            continue
        
        # 尝试将相对路径转换为绝对路径
        # 1. 首先尝试相对于Markdown文件的路径
        rel_path_to_md = os.path.join(md_dir, img_path)
        # 2. 然后尝试相对于工作空间根目录的路径
        rel_path_to_root = os.path.join(workspace_root, img_path)
        
        # 检查图片文件是否存在
        if os.path.exists(rel_path_to_md):
            # 图片在Markdown文件所在目录或子目录中
            # 转换为相对于工作空间根目录的路径
            abs_path = os.path.abspath(rel_path_to_md)
            # 转换为file:///格式的URL
            file_url = abs_path.replace('\\', '/')
            file_url = f"file:///{file_url}"
            
            # 替换原路径
            old_img_ref = f"![{alt_text}]({img_path})"
            new_img_ref = f"![{alt_text}]({file_url})"
            content = content.replace(old_img_ref, new_img_ref)
            changes_made = True
            print(f"{md_file_path}: 已修复图片引用: {img_path} -> {file_url}")
        elif os.path.exists(rel_path_to_root):
            # 图片在工作空间根目录或其他子目录中
            abs_path = os.path.abspath(rel_path_to_root)
            file_url = abs_path.replace('\\', '/')
            file_url = f"file:///{file_url}"
            
            old_img_ref = f"![{alt_text}]({img_path})"
            new_img_ref = f"![{alt_text}]({file_url})"
            content = content.replace(old_img_ref, new_img_ref)
            changes_made = True
            print(f"{md_file_path}: 已修复图片引用: {img_path} -> {file_url}")
        else:
            print(f"{md_file_path}: 警告: 找不到图片文件: {img_path}")
    
    # 如果有修改，则写回文件
    if changes_made:
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"{md_file_path}: 已更新并创建备份: {backup_path}")
    
    return changes_made


def main():
    parser = argparse.ArgumentParser(description='修复Markdown文件中的图片引用路径')
    parser.add_argument('--path', type=str, default=os.getcwd(), 
                        help='要处理的目录或Markdown文件路径，默认为当前目录')
    parser.add_argument('--dry-run', action='store_true', 
                        help='仅显示需要修改的内容，不实际修改文件')
    
    args = parser.parse_args()
    target_path = args.path
    
    # 确定工作空间根目录
    workspace_root = os.path.abspath(os.getcwd())
    print(f"工作空间根目录: {workspace_root}")
    
    if os.path.isfile(target_path) and target_path.lower().endswith('.md'):
        # 处理单个Markdown文件
        if not args.dry_run:
            fix_markdown_images(target_path, workspace_root)
        else:
            print(f"[模拟] 将会处理文件: {target_path}")
    elif os.path.isdir(target_path):
        # 处理目录下所有Markdown文件
        md_files = get_all_markdown_files(target_path)
        print(f"找到 {len(md_files)} 个Markdown文件")
        
        for md_file in md_files:
            if not args.dry_run:
                fix_markdown_images(md_file, workspace_root)
            else:
                print(f"[模拟] 将会处理文件: {md_file}")
    else:
        print(f"错误: {target_path} 不是有效的文件或目录")
        return
    
    print("\n处理完成!")
    print("注意:")
    print("1. 所有修改过的文件都创建了备份(.bak.时间戳)")
    print("2. 如果需要恢复原状，可以使用备份文件覆盖原文件")
    print("3. 修复后的图片引用使用了绝对路径，应该能在任何预览模式下正常显示")


if __name__ == "__main__":
    main()