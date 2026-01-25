import pandas as pd
import numpy as np
from scipy import signal
import os
from typing import Dict, List, Tuple
import json


class QTBFSScorer:
    """
    QTBFS康复评分系统 - 基于压电传感器信号的肌腱康复评分系统
    """
    
    def __init__(self):
        self.state0_data = {}  # 状态0（参照组）的数据
        self.current_data = {}  # 当前状态的数据
    
    def calculate_qtbfs_score(self, input_data: Dict) -> Dict:
        """
        计算QTBFS康复评分
        """
        # 这里应该从input_data加载文件，但现在我们使用模拟数据
        # 实际应用中，需要从文件路径加载数据
        try:
            # 模拟评分计算 - 实际应用中需要从input_data解析文件
            # 解析数据
            self._parse_input_data(input_data)
            
            # 计算各域得分
            domain_i_score = self._calculate_domain_i()
            domain_ii_score = self._calculate_domain_ii()
            domain_iii_score = self._calculate_domain_iii()
            
            # 计算总分
            total_score = domain_i_score['total'] + domain_ii_score['total'] + domain_iii_score['total']
            
            # 确定康复阶段
            stage = self._determine_stage(total_score)
            
            return {
                'domain_I': domain_i_score,
                'domain_II': domain_ii_score,
                'domain_III': domain_iii_score,
                'total_score': round(total_score, 2),
                'stage': stage,
                'stage_code': self._score_to_stage_code(total_score)
            }
        except Exception as e:
            # 如果计算失败，返回默认值
            return self._get_default_scores()
    
    def _parse_input_data(self, input_data: Dict):
        """
        解析输入数据
        """
        # 实际应用中，这里会加载CSV文件数据
        # 目前使用模拟数据
        pass
    
    def _calculate_domain_i(self) -> Dict:
        """
        计算域I：力学承载能力（满分40分）
        """
        # 模拟计算
        sub_a_score = np.random.randint(12, 21)  # 12-20分（峰值响应能力）
        sub_b_score = np.random.randint(6, 13)   # 6-12分（角度响应曲线）
        sub_c_score = np.random.randint(4, 9)    # 4-8分（大角度响应）
        
        total = sub_a_score + sub_b_score + sub_c_score
        
        return {
            'subA_score': sub_a_score,
            'subA_detail': {'avg_retention': round(np.random.uniform(0.7, 1.0), 4)},
            'subB_score': sub_b_score,
            'subB_detail': {'correlation_r': round(np.random.uniform(0.8, 1.0), 4)},
            'subC_score': sub_c_score,
            'subC_detail': {'retention_120': round(np.random.uniform(0.6, 1.0), 4)},
            'total': total
        }
    
    def _calculate_domain_ii(self) -> Dict:
        """
        计算域II：动态适应能力（满分35分）
        """
        # 模拟计算
        sub_a_score = np.random.randint(9, 16)  # 9-15分（速度梯度响应）
        sub_b_score = np.random.randint(7, 13)  # 7-12分（重复稳定性）
        sub_c_score = np.random.randint(4, 9)   # 4-8分（速度-角度交互）
        
        total = sub_a_score + sub_b_score + sub_c_score
        
        return {
            'subA_score': sub_a_score,
            'subA_detail': {'avg_consistency': round(np.random.uniform(0.7, 1.0), 4)},
            'subB_score': sub_b_score,
            'subB_detail': {'avg_cv': round(np.random.uniform(0.05, 0.15), 4)},
            'subC_score': sub_c_score,
            'subC_detail': {'avg_retention': round(np.random.uniform(0.7, 1.0), 4)},
            'total': total
        }
    
    def _calculate_domain_iii(self) -> Dict:
        """
        计算域III：功能储备能力（满分25分）
        """
        # 模拟计算
        sub_a_score = np.random.randint(6, 11)  # 6-10分（快速响应效率）
        sub_b_score = np.random.randint(4, 11)  # 4-10分（极限条件响应）
        sub_c_score = np.random.randint(3, 6)   # 3-5分（信号质量指数）
        
        total = sub_a_score + sub_b_score + sub_c_score
        
        return {
            'subA_score': sub_a_score,
            'subA_detail': {'avg_ratio': round(np.random.uniform(0.8, 1.1), 4)},
            'subB_score': sub_b_score,
            'subB_detail': {'extreme_retention': round(np.random.uniform(0.6, 1.0), 4)},
            'subC_score': sub_c_score,
            'subC_detail': {'snr_db': round(np.random.uniform(15, 25), 2)},
            'total': total
        }
    
    def _determine_stage(self, total_score: float) -> str:
        """
        根据总分确定康复阶段
        """
        if total_score >= 85:
            return "完全康复/正常"
        elif total_score >= 70:
            return "康复晚期"
        elif total_score >= 50:
            return "康复中期"
        elif total_score >= 30:
            return "康复早期"
        else:
            return "严重损伤"
    
    def _score_to_stage_code(self, total_score: float) -> int:
        """
        将分数转换为阶段代码
        """
        if total_score >= 85:
            return 0
        elif total_score >= 70:
            return 1
        elif total_score >= 50:
            return 2
        elif total_score >= 30:
            return 3
        else:
            return 4
    
    def _get_default_scores(self) -> Dict:
        """
        获取默认评分（错误时使用）
        """
        return {
            'domain_I': {
                'subA_score': 0, 'subA_detail': {},
                'subB_score': 0, 'subB_detail': {},
                'subC_score': 0, 'subC_detail': {},
                'total': 0
            },
            'domain_II': {
                'subA_score': 0, 'subA_detail': {},
                'subB_score': 0, 'subB_detail': {},
                'subC_score': 0, 'subC_detail': {},
                'total': 0
            },
            'domain_III': {
                'subA_score': 0, 'subA_detail': {},
                'subB_score': 0, 'subB_detail': {},
                'subC_score': 0, 'subC_detail': {},
                'total': 0
            },
            'total_score': 0,
            'stage': "数据错误",
            'stage_code': -1
        }


def extract_features_from_signal(data: np.ndarray, sampling_rate: float) -> Dict:
    """
    从信号中提取特征
    """
    if len(data) < 2:
        return {}
    
    # 基本统计特征
    features = {
        'peak': np.max(data),
        'min_val': np.min(data),
        'peak_to_peak': np.max(data) - np.min(data),
        'mean_val': np.mean(data),
        'std_val': np.std(data),
        'rms': np.sqrt(np.mean(data**2)),
        'area': np.trapz(data) / sampling_rate
    }
    
    # 上升时间和下降时间
    start_value = data[0] if len(data) > 0 else 0
    peak_value = features['peak']
    peak_index = np.argmax(data)
    
    # 计算上升时间
    threshold_10 = start_value + 0.1 * (peak_value - start_value)
    threshold_90 = start_value + 0.9 * (peak_value - start_value)
    
    try:
        idx_10 = np.where(data[:peak_index] >= threshold_10)[0]
        idx_90 = np.where(data[:peak_index] >= threshold_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = (idx_90[0] - idx_10[0]) / sampling_rate
            features['rise_time'] = rise_time
            features['rise_slope'] = (peak_value - start_value) / rise_time if rise_time > 0 else 0
        else:
            features['rise_time'] = 0
            features['rise_slope'] = 0
    except:
        features['rise_time'] = 0
        features['rise_slope'] = 0
    
    # 计算下降时间
    end_value = data[-1] if len(data) > 0 else 0
    threshold_10_fall = end_value + 0.1 * (peak_value - end_value)
    threshold_90_fall = end_value + 0.9 * (peak_value - end_value)
    
    try:
        post_peak_data = data[peak_index:]
        idx_90_fall = np.where(post_peak_data <= threshold_90_fall)[0]
        idx_10_fall = np.where(post_peak_data <= threshold_10_fall)[0]
        
        if len(idx_10_fall) > 0 and len(idx_90_fall) > 0:
            fall_time = (idx_10_fall[0] - idx_90_fall[0]) / sampling_rate
            features['fall_time'] = fall_time
            features['fall_slope'] = (peak_value - end_value) / fall_time if fall_time > 0 else 0
        else:
            features['fall_time'] = 0
            features['fall_slope'] = 0
    except:
        features['fall_time'] = 0
        features['fall_slope'] = 0
    
    # 对称性和峰值位置
    features['symmetry'] = min(features.get('rise_time', 0), features.get('fall_time', 0)) / \
                           max(features.get('rise_time', 1), features.get('fall_time', 1))
    features['peak_position'] = peak_index / len(data) if len(data) > 0 else 0
    
    return features