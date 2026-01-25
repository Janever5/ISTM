import re

def remove_duplicate_functions_and_fix_html():
    """
    修复前端HTML文件中的重复函数定义和结构问题
    """
    with open('waveform_miniprogram.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复重复的 selectedSplitFile 声明
    pattern = r"(let selectedSplitFile = null;\s*\n)+"
    content = re.sub(pattern, 'let selectedSplitFile = null;\n', content)

    # 修复重复函数定义问题
    # 首先找出所有重复函数的定义位置
    functions_to_fix = [
        'handleSplitFileSelect',
        'loadVisualizationData', 
        'renderVisualizationChart',
        'updateVisualization',
        'showNotification',
        'updateTrainingLog',
        'handleVisFileSelect'
    ]

    # 对于每个函数，保留第一次定义，后续的替换为注释
    for func_name in functions_to_fix:
        # 找到所有匹配的函数定义
        pattern = rf'(function\s+{func_name}\s*\([^)]*\)\s*\{{(?:[^{{}}]++|{{(?:[^{{}}]++|{{[^}}]*}})*}})*}})'
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        
        # 从第二个匹配项开始，替换为注释
        for i in range(len(matches)-1, 0, -1):  # 从后往前替换，避免偏移问题
            match = matches[i]
            # 用注释替代重复的函数定义
            replacement = f'/* Duplicate function definition removed: {func_name} */\n// {match.group(0).replace(chr(10), chr(10)+"// ")}'
            content = content[:match.start()] + replacement + content[match.end():]

    # 修复HTML结构问题 - 移除重复的html标签
    content = re.sub(r'<\/html>\s*<\/html>\s*$', '</html>', content, flags=re.MULTILINE)

    # 修复switchTab函数可能存在的问题 - 确保其定义正确
    # 查找switchTab函数定义
    switch_tab_pattern = r'(function switchTab\(tabName\) \{.*?)(\n\s*\};?\s*\n)'
    content = re.sub(switch_tab_pattern, r'\1\n        }\n', content, flags=re.DOTALL)

    # 写回修复后的内容
    with open('waveform_miniprogram_fixed.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("前端文件修复完成，已生成 waveform_miniprogram_fixed.html")

if __name__ == "__main__":
    remove_duplicate_functions_and_fix_html()