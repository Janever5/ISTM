import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, detrend
import os
import re
from typing import Dict, List, Tuple, Any

class QTBFSScorer:
    """
    QTBFS康复评分系统 - 真实实现版 (含可视化支持)
    基于压电传感器信号的肌腱康复评分系统
    """
    
    def __init__(self):
        # 评分标准速查表 (根据文档 第十章)
        self.SCORING_TABLES = {
            'domain_I_A': [(0.90, 20), (0.75, 16), (0.50, 12), (0.25, 8), (0.0, 4)],
            'domain_I_B': [(0.95, 12), (0.85, 9), (0.70, 6), (0.0, 3)],
            'domain_I_C': [(0.80, 8), (0.60, 6), (0.40, 4), (0.0, 2)],
            'domain_II_A': [(0.90, 15), (0.80, 12), (0.65, 9), (0.50, 6), (0.0, 3)],
            'domain_II_B': [(0.40, 2), (0.25, 4), (0.15, 7), (0.10, 10), (0.0, 12)], # CV越小越好
            'domain_III_A': [(0.90, 10), (0.80, 8), (0.65, 6), (0.50, 4), (0.0, 2)], 
            'domain_III_B': [(0.80, 10), (0.60, 7), (0.40, 4), (0.0, 2)],
            'domain_III_C': [(20, 5), (15, 4), (10, 3), (0, 1)]
        }

    def calculate_qtbfs_score(self, input_data: Dict) -> Dict:
        """
        主入口：计算QTBFS康复评分
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
            
            # 4. 提取可视化数据 (仅保留 visualization 字段，不混入特征计算)
            vis_data = {
                'state0': {k: v['vis_data'] for k, v in state0_feats.items()},
                'current': {k: v['vis_data'] for k, v in current_feats.items()}
            }
            
            return {
                'domain_I': d1_res,
                'domain_II': d2_res,
                'domain_III': d3_res,
                'total_score': round(total, 1),
                'stage': stage,
                'stage_code': stage_code,
                'visualizations': vis_data # 新增：用于前端绘图
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._error_result(f"计算出错: {str(e)}")

    # --- 数据处理核心 (含可视化数据提取) ---

    def _process_batch(self, file_map: Dict[str, str]) -> Dict[str, Any]:
        """批量处理文件，提取特征及可视化数据"""
        features = {}
        for key, path in file_map.items():
            try:
                # 读取数据
                if path.endswith('.csv'):
                    try:
                        df = pd.read_csv(path, encoding='utf-8')
                    except:
                        df = pd.read_csv(path, encoding='gbk')
                else:
                    df = pd.read_excel(path)
                
                # 寻找数据列
                data_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
                raw_signal = pd.to_numeric(df[data_col], errors='coerce').fillna(0).values
                
                # --- 预处理流程 ---
                # 1. 去趋势 (Detrend)
                detrended = signal.detrend(raw_signal, type='linear')
                # 2. 滤波
                filtered = np.convolve(detrended, np.ones(5)/5, mode='same')
                # 3. 基线归零 (使用第5百分位)
                baseline = np.percentile(filtered, 5)
                corrected = np.maximum(0, filtered - baseline)
                
                # --- 特征提取 ---
                peaks, _ = find_peaks(corrected, height=np.max(corrected)*0.20, distance=50)
                if len(peaks) == 0:
                    peak_vals = np.array([np.max(corrected)])
                else:
                    peak_vals = corrected[peaks]
                
                sig_power = np.mean(peak_vals**2) if len(peak_vals) > 0 else 0
                noise_mask = corrected < (np.max(corrected) * 0.1)
                noise_vals = corrected[noise_mask]
                noise_power = np.std(noise_vals)**2 if len(noise_vals) > 10 else 1e-6
                
                # --- 可视化数据准备 (降采样以减少传输量) ---
                # 目标：每个文件约 500-800 个点
                step = max(1, len(raw_signal) // 800)
                
                # 原始信号 (为了对比，也可以减去均值以便同框显示，或者直接显示原始)
                # 这里我们显示原始信号的去均值版本，方便与处理后信号对比趋势
                raw_display = raw_signal - np.mean(raw_signal)
                
                features[key] = {
                    'peak_mean': np.mean(peak_vals),
                    'peak_std': np.std(peak_vals),
                    'peak_cv': (np.std(peak_vals) / np.mean(peak_vals)) if np.mean(peak_vals) > 1e-6 else 0,
                    'signal_power': sig_power,
                    'noise_power': noise_power + 1e-9,
                    # 新增：可视化数据
                    'vis_data': {
                        'raw': raw_display[::step].tolist(),
                        'processed': corrected[::step].tolist(),
                        'labels': list(range(0, len(raw_display), step))
                    }
                }
            except Exception as e:
                print(f"处理文件 {key} 失败: {e}")
                continue
        return features

    # --- 评分逻辑 (保持不变) ---

    def _get_score(self, value, rule_key):
        rules = self.SCORING_TABLES[rule_key]
        if rule_key == 'domain_II_B':
            for thresh, score in rules:
                if value > thresh: return score
            return 12
        for thresh, score in rules:
            if value >= thresh: return score
        return 0

    def _calc_domain_I(self, curr, ref) -> Dict:
        angles = ['angle_30', 'angle_45', 'angle_60', 'angle_90']
        ratios = []
        for a in angles:
            if a in curr and a in ref and ref[a]['peak_mean'] > 1e-6:
                ratios.append(curr[a]['peak_mean'] / ref[a]['peak_mean'])
        avg_retention = np.mean(ratios) if ratios else 0
        subA = self._get_score(avg_retention, 'domain_I_A')

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

        ret_120 = 0
        if 'angle_120' in curr and 'angle_120' in ref and ref['angle_120']['peak_mean'] > 1e-6:
            ret_120 = curr['angle_120']['peak_mean'] / ref['angle_120']['peak_mean']
        subC = self._get_score(ret_120, 'domain_I_C')

        return {
            'total': subA + subB + subC,
            'subA_score': subA, 'subA_detail': {'avg_retention': avg_retention},
            'subB_score': subB, 'subB_detail': {'correlation_r': corr},
            'subC_score': subC, 'subC_detail': {'retention_120': ret_120}
        }

    def _calc_domain_II(self, curr, ref) -> Dict:
        consistencies = []
        for deg in ['30', '60', '90']:
            k1 = f"speed_{deg}_1s"
            k3 = f"speed_{deg}_3s"
            if k1 in curr and k3 in curr and curr[k3]['peak_mean'] > 1e-6:
                cons = 1 - abs(curr[k1]['peak_mean'] - curr[k3]['peak_mean']) / curr[k3]['peak_mean']
                consistencies.append(cons)
        avg_cons = np.mean(consistencies) if consistencies else 0
        subA = self._get_score(avg_cons, 'domain_II_A')

        cvs = [v['peak_cv'] for k, v in curr.items() if k.startswith('speed_')]
        avg_cv = np.mean(cvs) if cvs else 0.5 
        subB = self._get_score(avg_cv, 'domain_II_B')

        speed_ratios = []
        for k in curr:
            if k.startswith('speed_') and k in ref and ref[k]['peak_mean'] > 1e-6:
                speed_ratios.append(curr[k]['peak_mean'] / ref[k]['peak_mean'])
        avg_speed_ret = np.mean(speed_ratios) if speed_ratios else 0
        subC = int(avg_speed_ret * 8) if avg_speed_ret <= 1 else 8

        return {
            'total': subA + subB + subC,
            'subA_score': subA, 'subA_detail': {'avg_consistency': avg_cons},
            'subB_score': subB, 'subB_detail': {'avg_cv': avg_cv},
            'subC_score': subC, 'subC_detail': {'avg_retention': avg_speed_ret}
        }

    def _calc_domain_III(self, curr, ref) -> Dict:
        ratios = []
        for deg in ['30', '60', '90']:
            k1 = f"speed_{deg}_1s"
            k3 = f"speed_{deg}_3s"
            if k1 in curr and k3 in curr and curr[k3]['peak_mean'] > 1e-6:
                ratios.append(curr[k1]['peak_mean'] / curr[k3]['peak_mean'])
        avg_ratio = np.mean(ratios) if ratios else 0
        if 0.9 <= avg_ratio <= 1.1: subA = 10
        elif 0.8 <= avg_ratio < 0.9 or 1.1 < avg_ratio <= 1.2: subA = 8
        elif 0.65 <= avg_ratio < 0.8: subA = 6
        elif 0.5 <= avg_ratio < 0.65: subA = 4
        else: subA = 2

        extreme_keys = ['speed_90_1s', 'angle_120']
        ext_ratios = []
        for k in extreme_keys:
            if k in curr and k in ref and ref[k]['peak_mean'] > 1e-6:
                ext_ratios.append(curr[k]['peak_mean'] / ref[k]['peak_mean'])
        ext_ret = np.mean(ext_ratios) if ext_ratios else 0
        subB = self._get_score(ext_ret, 'domain_III_B')

        snr_vals = []
        for v in curr.values():
            if v['noise_power'] > 1e-9:
                snr_vals.append(10 * np.log10(v['signal_power']/v['noise_power']))
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