import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import os
import re
from typing import Dict, List, Tuple, Any

class QTBFSScorer:
    """
    QTBFS康复评分系统 - 真实实现版
    基于压电传感器信号的肌腱康复评分系统
    """
    
    def __init__(self):
        # 评分标准速查表 (根据文档 第十章)
        self.SCORING_TABLES = {
            'domain_I_A': [(0.90, 20), (0.75, 16), (0.50, 12), (0.25, 8), (0.0, 4)],
            'domain_I_B': [(0.95, 12), (0.85, 9), (0.70, 6), (0.0, 3)],
            'domain_I_C': [(0.80, 8), (0.60, 6), (0.40, 4), (0.0, 2)],
            'domain_II_A': [(0.90, 15), (0.80, 12), (0.65, 9), (0.50, 6), (0.0, 3)],
            'domain_II_B': [(0.40, 2), (0.25, 4), (0.15, 7), (0.10, 10), (0.0, 12)], # 注意这是越小越好 (CV)
            'domain_III_A': [(0.90, 10), (0.80, 8), (0.65, 6), (0.50, 4), (0.0, 2)], # 需特殊处理 1.10区间
            'domain_III_B': [(0.80, 10), (0.60, 7), (0.40, 4), (0.0, 2)],
            'domain_III_C': [(20, 5), (15, 4), (10, 3), (0, 1)]
        }

    def calculate_qtbfs_score(self, input_data: Dict) -> Dict:
        """
        主入口：计算QTBFS康复评分
        input_data 结构: {'state0': {'angle_30': path, ...}, 'current_state': {'angle_30': path, ...}}
        """
        try:
            # 1. 加载并预处理所有数据
            state0_feats = self._process_batch(input_data.get('state0', {}))
            current_feats = self._process_batch(input_data.get('current_state', {}))
            
            if not current_feats:
                return self._error_result("未检测到有效的当前状态数据")

            # 2. 计算各域得分
            d1_res = self._calc_domain_I(current_feats, state0_feats)
            d2_res = self._calc_domain_II(current_feats, state0_feats)
            d3_res = self._calc_domain_III(current_feats, state0_feats)
            
            # 3. 汇总
            total = d1_res['total'] + d2_res['total'] + d3_res['total']
            stage, stage_code = self._determine_stage(total)
            
            return {
                'domain_I': d1_res,
                'domain_II': d2_res,
                'domain_III': d3_res,
                'total_score': round(total, 1),
                'stage': stage,
                'stage_code': stage_code
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._error_result(f"计算出错: {str(e)}")

    # --- 数据处理核心 ---

    def _process_batch(self, file_map: Dict[str, str]) -> Dict[str, Any]:
        """批量处理文件，提取每个文件的特征"""
        features = {}
        for key, path in file_map.items():
            try:
                # 读取CSV (兼容 backend_server 的保存路径)
                df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                
                # 寻找数据列 (第3列 or 名称匹配)
                data_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
                raw_signal = pd.to_numeric(df[data_col], errors='coerce').fillna(0).values
                
                # 预处理 (文档 4.2)
                # 1. 简单滤波 (移动平均)
                filtered = np.convolve(raw_signal, np.ones(5)/5, mode='same')
                # 2. 基线校正 (前100点)
                baseline = np.mean(filtered[:100]) if len(filtered) > 100 else np.min(filtered)
                corrected = np.maximum(0, filtered - baseline) # 确保非负
                
                # 特征提取 (文档 5.4 多周期统计)
                # 简单峰值检测
                peaks, _ = find_peaks(corrected, height=np.max(corrected)*0.3, distance=50)
                if len(peaks) == 0:
                    peak_vals = [np.max(corrected)] # 兜底
                else:
                    peak_vals = corrected[peaks]
                
                features[key] = {
                    'peak_mean': np.mean(peak_vals),
                    'peak_std': np.std(peak_vals),
                    'peak_cv': (np.std(peak_vals) / np.mean(peak_vals)) if np.mean(peak_vals) > 0 else 0,
                    'signal_power': np.mean(peak_vals**2), # 简化功率
                    'noise_power': np.std(corrected[:100]) + 1e-6 # 防止除零
                }
            except Exception as e:
                print(f"处理文件 {key} 失败: {e}")
                continue
        return features

    # --- 评分逻辑 (文档 6.0) ---

    def _get_score(self, value, rule_key):
        """查表计分通用函数"""
        rules = self.SCORING_TABLES[rule_key]
        
        # 特殊处理 CV (越小越好)
        if rule_key == 'domain_II_B':
            for thresh, score in rules:
                if value > thresh: return score # 比如 > 0.4 得 2分
            return 12 # < 0.1 得 12分
            
        # 标准处理 (越大越好)
        for thresh, score in rules:
            if value >= thresh: return score
        return 0

    def _calc_domain_I(self, curr, ref) -> Dict:
        """域I：力学承载能力 (Ref: 文档 6.1)"""
        # A: 峰值响应 (30,45,60,90)
        angles = ['angle_30', 'angle_45', 'angle_60', 'angle_90']
        ratios = []
        for a in angles:
            if a in curr and a in ref and ref[a]['peak_mean'] > 0:
                ratios.append(curr[a]['peak_mean'] / ref[a]['peak_mean'])
        
        avg_retention = np.mean(ratios) if ratios else 0
        subA = self._get_score(avg_retention, 'domain_I_A')

        # B: 角度曲线相关性 (简化为计算可用角度的Correlation)
        # 提取共同存在的角度键
        common_angles = sorted([k for k in curr.keys() if k.startswith('angle_') and k in ref])
        if len(common_angles) >= 3:
            curr_vec = [curr[k]['peak_mean'] for k in common_angles]
            ref_vec = [ref[k]['peak_mean'] for k in common_angles]
            if np.std(curr_vec) > 0 and np.std(ref_vec) > 0:
                corr = np.corrcoef(curr_vec, ref_vec)[0, 1]
            else:
                corr = 0
        else:
            corr = 0
        subB = self._get_score(corr, 'domain_I_B')

        # C: 大角度 (120)
        ret_120 = 0
        if 'angle_120' in curr and 'angle_120' in ref:
            ret_120 = curr['angle_120']['peak_mean'] / ref['angle_120']['peak_mean']
        subC = self._get_score(ret_120, 'domain_I_C')

        return {
            'total': subA + subB + subC,
            'subA_score': subA, 'subA_detail': {'avg_retention': avg_retention},
            'subB_score': subB, 'subB_detail': {'correlation_r': corr},
            'subC_score': subC, 'subC_detail': {'retention_120': ret_120}
        }

    def _calc_domain_II(self, curr, ref) -> Dict:
        """域II：动态适应能力 (Ref: 文档 6.2)"""
        # A: 速度梯度一致性 (30,60,90 的 1s vs 3s)
        consistencies = []
        for deg in ['30', '60', '90']:
            k1 = f"speed_{deg}_1s"
            k3 = f"speed_{deg}_3s"
            if k1 in curr and k3 in curr and curr[k3]['peak_mean'] > 0:
                cons = 1 - abs(curr[k1]['peak_mean'] - curr[k3]['peak_mean']) / curr[k3]['peak_mean']
                consistencies.append(cons)
        
        avg_cons = np.mean(consistencies) if consistencies else 0
        subA = self._get_score(avg_cons, 'domain_II_A')

        # B: 重复稳定性 (CV) - 使用所有speed文件
        cvs = [v['peak_cv'] for k, v in curr.items() if k.startswith('speed_')]
        avg_cv = np.mean(cvs) if cvs else 0.5 # 默认给个差值
        subB = self._get_score(avg_cv, 'domain_II_B')

        # C: 交互保留率 (简化)
        # 比较所有对应速度文件的比值均值
        speed_ratios = []
        for k in curr:
            if k.startswith('speed_') and k in ref:
                speed_ratios.append(curr[k]['peak_mean'] / ref[k]['peak_mean'])
        avg_speed_ret = np.mean(speed_ratios) if speed_ratios else 0
        subC = int(avg_speed_ret * 8) if avg_speed_ret <= 1 else 8 # 满分8

        return {
            'total': subA + subB + subC,
            'subA_score': subA, 'subA_detail': {'avg_consistency': avg_cons},
            'subB_score': subB, 'subB_detail': {'avg_cv': avg_cv},
            'subC_score': subC, 'subC_detail': {'avg_retention': avg_speed_ret}
        }

    def _calc_domain_III(self, curr, ref) -> Dict:
        """域III：功能储备 (Ref: 文档 6.3)"""
        # A: 快速响应 (1s/3s)
        # 逻辑同Domain II A，但这里是比值
        ratios = []
        for deg in ['30', '60', '90']:
            k1 = f"speed_{deg}_1s"
            k3 = f"speed_{deg}_3s"
            if k1 in curr and k3 in curr and curr[k3]['peak_mean'] > 0:
                ratios.append(curr[k1]['peak_mean'] / curr[k3]['peak_mean'])
        
        avg_ratio = np.mean(ratios) if ratios else 0
        # 查表逻辑特殊：0.9-1.1满分
        if 0.9 <= avg_ratio <= 1.1: subA = 10
        elif 0.8 <= avg_ratio < 0.9 or 1.1 < avg_ratio <= 1.2: subA = 8
        elif 0.65 <= avg_ratio < 0.8: subA = 6
        elif 0.5 <= avg_ratio < 0.65: subA = 4
        else: subA = 2

        # B: 极限条件 (90-1s 和 120度)
        extreme_keys = ['speed_90_1s', 'angle_120']
        ext_ratios = []
        for k in extreme_keys:
            if k in curr and k in ref:
                ext_ratios.append(curr[k]['peak_mean'] / ref[k]['peak_mean'])
        ext_ret = np.mean(ext_ratios) if ext_ratios else 0
        subB = self._get_score(ext_ret, 'domain_III_B')

        # C: 信噪比 (取任意一个文件的SNR)
        snr_vals = [10 * np.log10(v['signal_power']/v['noise_power']) for v in curr.values()]
        avg_snr = np.mean(snr_vals) if snr_vals else 0
        subC = self._get_score(avg_snr, 'domain_III_C')

        return {
            'total': subA + subB + subC,
            'subA_score': subA, 'subA_detail': {'avg_ratio': avg_ratio},
            'subB_score': subB, 'subB_detail': {'extreme_retention': ext_ret},
            'subC_score': subC, 'subC_detail': {'snr_db': avg_snr}
        }

    def _determine_stage(self, total):
        if total >= 85: return "完全康复/正常", 0
        if total >= 70: return "康复晚期", 1
        if total >= 50: return "康复中期", 2
        if total >= 30: return "康复早期", 3
        return "严重损伤", 4

    def _error_result(self, msg):
        return {
            'total_score': 0, 'stage': msg, 'stage_code': -1,
            'domain_I': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}},
            'domain_II': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}},
            'domain_III': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}}
        }