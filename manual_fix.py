def manual_clean():
    """
    手动清理重复定义
    """
    with open('waveform_miniprogram.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复 selectedSplitFile 的重复声明
    content = content.replace(
        'let selectedSplitFile = null;',
        '// let selectedSplitFile = null;  // Already declared earlier'
    )
    # 然后恢复第一次声明
    first_declare_pos = content.find('// let selectedSplitFile = null;  // Already declared earlier')
    if first_declare_pos != -1:
        # 找到第一次出现的位置并恢复
        content = content.replace(
            content[first_declare_pos:first_declare_pos+70],
            'let selectedSplitFile = null;'
        , 1)

    # 定位到特定的重复函数定义并删除
    lines = content.split('\n')
    
    # 创建新的行列表，只保留第一次出现的函数定义
    new_lines = []
    encountered_funcs = {}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是需要去重的函数定义
        is_duplicate = False
        
        if 'function handleSplitFileSelect(' in line:
            if 'handleSplitFileSelect' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['handleSplitFileSelect'] = True
        
        elif 'function loadVisualizationData(' in line:
            if 'loadVisualizationData' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['loadVisualizationData'] = True
        
        elif 'function renderVisualizationChart(' in line:
            if 'renderVisualizationChart' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['renderVisualizationChart'] = True
        
        elif 'function updateVisualization(' in line:
            if 'updateVisualization' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['updateVisualization'] = True
        
        elif 'function showNotification(message, type' in line and 'showNotification' not in encountered_funcs:
            # 这是一个特殊的case，因为有多个不同签名的showNotification
            # 我们只保留第一个
            encountered_funcs['showNotification'] = True
        
        elif 'function showNotification(' in line and 'showNotification' in encountered_funcs:
            # 跳过后续的showNotification定义
            brace_count = 0
            j = i
            while j < len(lines):
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                j += 1
                if brace_count <= 0 and '}' in lines[j-1]:
                    break
            i = j
            continue
        
        elif 'function updateTrainingLog(' in line:
            if 'updateTrainingLog' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['updateTrainingLog'] = True
        
        elif 'function handleVisFileSelect(' in line:
            if 'handleVisFileSelect' in encountered_funcs:
                is_duplicate = True
                # 跳过整个函数定义
                brace_count = 0
                j = i
                while j < len(lines):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    j += 1
                    if brace_count <= 0 and '}' in lines[j-1]:
                        break
                i = j
                continue
            else:
                encountered_funcs['handleVisFileSelect'] = True
        
        if not is_duplicate:
            new_lines.append(line)
        i += 1

    # 修复HTML结构问题 - 移除多余的html结束标签
    fixed_content = '\n'.join(new_lines)
    parts = fixed_content.rsplit('</html>', 2)  # 从右边分割，保留最后一个
    if len(parts) > 1:
        fixed_content = ''.join(parts[:-1]) + '</html>'
    
    # 写入修复后的文件
    with open('waveform_miniprogram_cleaned.html', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("手动清理完成，已生成 waveform_miniprogram_cleaned.html")


if __name__ == "__main__":
    manual_clean()