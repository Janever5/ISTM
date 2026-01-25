# 测试各个模块是否可以正常工作
print("测试各个模块...")

try:
    from qtbfs_scorer import QTBFSScorer
    scorer = QTBFSScorer()
    result = scorer.calculate_qtbfs_score({"input_data": {}})
    print("✓ QTBFS模块正常")
except Exception as e:
    print(f"✗ QTBFS模块错误: {e}")
    import traceback
    traceback.print_exc()

try:
    from signal_splitter import SignalSplitter
    splitter = SignalSplitter()
    print("✓ SignalSplitter模块正常")
except Exception as e:
    print(f"✗ SignalSplitter模块错误: {e}")
    import traceback
    traceback.print_exc()

try:
    import json
    test_data = {"input_data": {}}
    json_str = json.dumps(test_data)
    parsed = json.loads(json_str)
    print("✓ JSON处理正常")
except Exception as e:
    print(f"✗ JSON处理错误: {e}")
    import traceback
    traceback.print_exc()

try:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    arr = np.array([1, 2, 3])
    print("✓ Pandas和NumPy正常")
except Exception as e:
    print(f"✗ Pandas/NumPy错误: {e}")
    import traceback
    traceback.print_exc()

print("模块测试完成")