def final_fix():
    """
    最终修复脚本 - 精确处理每一个重复定义
    """
    with open('waveform_miniprogram.html', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 定义需要处理的函数和它们的所有出现位置（行号从1开始）
    functions_to_fix = {
        'handleSplitFileSelect': [2184, 2384, 2577],  # 保留第一个，注释掉其余
        'loadVisualizationData': [2197, 2396],
        'renderVisualizationChart': [2253, 2452],
        'updateVisualization': [2317, 2516],
        'showNotification': [1458, 2322],
        'updateTrainingLog': [1641, 2334],
        'handleVisFileSelect': [1611, 2370],
        'selectedSplitFile': [1453, 2181, 2381]  # 处理重复的变量声明
    }

    # 注释掉重复的函数定义和变量声明（保留第一个）
    for func_name, positions in functions_to_fix.items():
        if len(positions) > 1:
            for pos in positions[1:]:  # 保留第一个，注释掉其余
                # 找到函数定义的开始行
                line_idx = pos - 1  # 转换为0索引
                if line_idx < len(lines):
                    # 注释掉整个多行函数定义
                    if 'function' in lines[line_idx] or 'let selectedSplitFile' in lines[line_idx]:
                        # 找到函数定义的结束（寻找匹配的闭合大括号）
                        lines[line_idx] = f"// REMOVED DUPLICATE {func_name}: {lines[line_idx]}"
                        
                        # 对于函数定义，通常需要注释多行直到匹配的闭合大括号
                        brace_count = 0
                        for i in range(line_idx, min(line_idx + 50, len(lines))):  # 限制搜索范围
                            if '{' in lines[i]:
                                brace_count += lines[i].count('{')
                            if '}' in lines[i]:
                                brace_count -= lines[i].count('}')
                            
                            if brace_count <= 0 and '}' in lines[i]:
                                # 找到了闭合大括号，注释从开始到结束的所有行
                                for j in range(line_idx + 1, i + 1):
                                    if j < len(lines):
                                        lines[j] = f"// {lines[j]}"
                                break
                            elif i > line_idx:
                                lines[i] = f"// {lines[i]}"

    # 修复HTML结构问题 - 确保只有一个结束标签
    # 找到最后几个行，确保HTML结构正确
    for i in range(len(lines)-1, max(0, len(lines)-20), -1):
        if '</html>' in lines[i] and lines[i].strip() != '</html>':
            # 如果html标签不在行首，规范化它
            if lines[i].strip() == '</html>':
                continue
            else:
                lines[i] = '</html>\n'

    # 确保文件以正确的HTML结束标签结尾
    html_end_found = False
    for i in range(len(lines)-1, max(0, len(lines)-10), -1):
        if '</html>' in lines[i]:
            # 确保这是最后一个HTML结束标签
            for j in range(i+1, len(lines)):
                if '</html>' in lines[j]:
                    lines[j] = ''  # 清空多余的结束标签
            html_end_found = True
            break
    
    # 如果没有找到合适的结束标签，添加一个
    if not html_end_found:
        lines.extend(['\n</body>\n', '</html>\n'])

    # 写入修复后的文件
    with open('waveform_miniprogram_final_fixed.html', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("最终修复完成，已生成 waveform_miniprogram_final_fixed.html")


if __name__ == "__main__":
    final_fix()