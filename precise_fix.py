import re

def fix_html_structure_and_duplicates():
    """
    更精确地修复HTML结构和重复函数定义
    """
    with open('waveform_miniprogram.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复重复的 selectedSplitFile 声明
    pattern = r'(let selectedSplitFile = null;\s*\n)+'
    content = re.sub(pattern, '        // 保存选中的分割文件\n        let selectedSplitFile = null;\n', content)

    # 找到所有函数定义的起始位置
    function_patterns = [
        r'function handleSplitFileSelect\(.*?\).*?\{.*?\}',
        r'function loadVisualizationData\(.*?\).*?\{.*?\}',
        r'function renderVisualizationChart\(.*?\).*?\{.*?\}',
        r'function updateVisualization\(.*?\).*?\{.*?\}',
        r'function showNotification\(.*?\).*?\{.*?\}',
        r'function updateTrainingLog\(.*?\).*?\{.*?\}',
        r'function handleVisFileSelect\(.*?\).*?\{.*?\}',
    ]
    
    # 使用正则表达式找到所有函数定义
    for func_pattern in function_patterns:
        # 使用非贪婪匹配找到所有函数定义
        matches = list(re.finditer(func_pattern, content, re.DOTALL))
        if len(matches) > 1:
            # 保留第一个，将其余的替换为注释
            for i in range(len(matches)-1, 0, -1):  # 从后往前替换
                match = matches[i]
                # 获取函数定义的完整内容
                func_def = match.group(0)
                # 替换为注释
                commented_func = f'        /* Removed duplicate function definition */\n        // {func_def.replace(chr(10), chr(10)+"        // ")}\n'
                content = content[:match.start()] + commented_func + content[match.end():]

    # 修复HTML结构问题 - 移除文件末尾多余的html标签
    content = re.sub(r'<\/html>\s*<\/html>\s*$', '</html>\n', content, flags=re.MULTILINE)

    # 确保只有一个完整的HTML结束标签
    html_end_pattern = r'(</body>\s*</html>.*$)'
    match = re.search(html_end_pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        end_part = match.group(1)
        # 确保结尾正确
        correct_end = '\n</body>\n</html>\n'
        content = content[:match.start()] + correct_end

    # 写回修复后的内容
    with open('waveform_miniprogram_fixed.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("前端文件精确修复完成，已生成 waveform_miniprogram_fixed.html")


def fix_with_manual_positions():
    """
    使用手动定位修复重复定义
    """
    with open('waveform_miniprogram.html', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 记录要删除的行范围
    lines_to_remove = []

    # 重复定义的位置
    duplicate_positions = [
        (2383, 2420),  # 第二个handleSplitFileSelect定义
        (2580, 2590),  # 第三个handleSplitFileSelect定义
        (2395, 2430),  # 第二个loadVisualizationData定义
        (2451, 2480),  # 第二个renderVisualizationChart定义
        (2515, 2525),  # 第二个updateVisualization定义
        (2321, 2330),  # 第二个showNotification定义
        (2333, 2340),  # 第二个updateTrainingLog定义
        (2369, 2375),  # 第二个handleVisFileSelect定义
    ]

    # 将要删除的行标记
    for start, end in duplicate_positions:
        for i in range(start-1, min(end, len(lines))):  # 转换为0索引
            if i < len(lines):
                lines[i] = f"// REMOVED DUPLICATE: {lines[i]}"

    # 写入修复后的文件
    with open('waveform_miniprogram_fixed_manual.html', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("手动修复完成，已生成 waveform_miniprogram_fixed_manual.html")


if __name__ == "__main__":
    print("执行精确修复...")
    fix_html_structure_and_duplicates()
    print("执行手动定位修复...")
    fix_with_manual_positions()