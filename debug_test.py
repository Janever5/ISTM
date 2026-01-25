from qtbfs_scorer import QTBFSScorer

try:
    scorer = QTBFSScorer()
    result = scorer.calculate_qtbfs_score({"input_data": {}})
    print("QTBFS评分计算成功:")
    print(result)
except Exception as e:
    print(f"QTBFS评分计算出错: {e}")
    import traceback
    traceback.print_exc()