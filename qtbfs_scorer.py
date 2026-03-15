import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import os
from typing import Dict, List, Tuple, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QTBFSScorer:
    """
    QTBFS康复评分系统 - 遵照《传感.md》规范的完整实现
    """
    
    def __init__(self):
        # 评分标准速查表 (根据文档 第十章)
        self.SCORING_TABLES = {
            'domain_I_A': [(0.90, 20), (0.75, 16), (0.50, 12), (0.25, 8), (0.0, 4)],
            'domain_I_B': [(0.95, 12), (0.85, 9), (0.70, 6), (0.0, 3)],
            'domain_I_C': [(0.80, 8), (0.60, 6), (0.40, 4), (0.0, 2)],
            'domain_II_A': [(0.90, 15), (0.80, 12), (0.65, 9), (0.50, 6), (0.0, 3)],
            'domain_II_B': [(0.40, 2), (0.25, 4), (0.15, 7), (0.10, 10), (0.0, 12)], # CV越小越好，倒序查表
            'domain_III_A': [ # 特殊区间逻辑
                {'range': (0.90, 1.10), 'score': 10},
                {'range': (0.80, 0.89), 'score': 8},
                {'range': (1.11, 1.20), 'score': 8},
                {'range': (0.65, 0.79), 'score': 6},
                {'range': (0.50, 0.64), 'score': 4},
            ],
            'domain_III_B': [(0.80, 10), (0.60, 7), (0.40, 4), (0.0, 2)],
            'domain_III_C': [(20, 5), (15, 4), (10, 3), (0, 1)]
        }

    def _butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
        """巴特沃斯带通滤波器"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # SciPy < 1.2.0 requires Wn to be a scalar for btype='band'
        try:
            b, a = signal.butter(order, [low, high], btype='band')
        except ValueError: # Fallback for older scipy
            b, a = signal.butter(order, high, btype='high')
            b, a = signal.butter(order, low, btype='low', a=a)
        y = signal.lfilter(b, a, data)
        return y

    def _get_score(self, value: float, rule_key: str) -> float:
        """通用查表评分函数"""
        rules = self.SCORING_TABLES[rule_key]
        
        # CV是反向指标，值越小分越高
        if rule_key == 'domain_II_B':
            for thresh, score in rules:
                if value > thresh:
                    return score
            return 12 # CV < 10%
        
        # 域III-A是区间查找
        if rule_key == 'domain_III_A':
            for rule in rules:
                low, high = rule['range']
                if low <= value <= high:
                    return rule['score']
            return 2 # 不在任何良好区间，得最低分
        
        # 其他都是正向指标，值越大分越高
        for thresh, score in rules:
            if value >= thresh:
                return score
        return 0
        
    def _extract_cycle_features(self, cycle_sig: np.ndarray, sampling_rate: float) -> Dict:
        """对单个周期信号提取所有特征(文档5.1)"""
        if len(cycle_sig) < 3: return None
        
        peak_idx = np.argmax(cycle_sig)
        peak_val = cycle_sig[peak_idx]
        min_val = np.min(cycle_sig)
        
        # 上升段 & 下降段
        rise_sig = cycle_sig[:peak_idx+1]
        fall_sig = cycle_sig[peak_idx:]
        
        # 上升/下降时间
        rise_time = (np.where(rise_sig >= peak_val*0.9)[0][0] - np.where(rise_sig >= peak_val*0.1)[0][0]) / sampling_rate if len(rise_sig) > 1 else 0
        fall_time = (np.where(fall_sig <= peak_val*0.9)[0][-1] - np.where(fall_sig <= peak_val*0.1)[0][-1]) / sampling_rate if len(fall_sig) > 1 else 0
        rise_time = max(rise_time, 1e-6) # 避免除零
        fall_time = max(fall_time, 1e-6)

        features = {
            'peak': peak_val,
            'min_val': min_val,
            'peak_to_peak': peak_val - min_val,
            'mean_val': np.mean(cycle_sig),
            'std_val': np.std(cycle_sig),
            'rms': np.sqrt(np.mean(cycle_sig**2)),
            'area': np.trapz(cycle_sig) / sampling_rate,
            'rise_time': rise_time,
            'fall_time': fall_time,
            'rise_slope': (peak_val - cycle_sig[0]) / rise_time,
            'fall_slope': (cycle_sig[-1] - peak_val) / fall_time,
            'symmetry': min(rise_time, fall_time) / max(rise_time, fall_time),
            'peak_position': peak_idx / len(cycle_sig)
        }
        return features

    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        完整处理单个文件，严格遵循《传感.md》第四、五章流程
        """
        # 步骤1: 读取文件
        try:
            if file_path.endswith('.csv'):
                try: df = pd.read_csv(file_path, encoding='utf-8')
                except: df = pd.read_csv(file_path, encoding='gbk')
            else:
                df = pd.read_excel(file_path)
            
            if len(df.columns) < 2: raise ValueError("文件列数不足")
            time_col, data_col = df.columns[1], df.columns[2]
            time_vals = pd.to_numeric(df[time_col], errors='coerce').fillna(0).values
            raw_signal = pd.to_numeric(df[data_col], errors='coerce').fillna(0).values
        except Exception as e:
            logging.error(f"读取文件 {os.path.basename(file_path)} 失败: {e}")
            return None

        # 步骤2: 计算采样率
        sampling_rate = 1.0 / (time_vals[1] - time_vals[0]) if len(time_vals) > 1 and time_vals[1] > time_vals[0] else 1000.0

        # 步骤3: 信号滤波
        filtered = self._butter_bandpass_filter(raw_signal, 0.1, 50, sampling_rate, order=4) if sampling_rate > 100 else signal.detrend(raw_signal)

        # 步骤4: 基线校正 (核心)
        baseline = np.mean(filtered[:100]) if len(filtered) > 100 else np.mean(filtered)
        corrected = filtered - baseline

        # 步骤5: 信号分割 (峰值检测法)
        peaks, _ = find_peaks(corrected, height=np.max(corrected)*0.3, distance=int(sampling_rate * 0.5)) # 假设周期>0.5s
        if len(peaks) < 2:
            logging.warning(f"文件 {os.path.basename(file_path)} 未检测到足够周期，将整个信号视为一个周期")
            cycles = [corrected]
        else:
            cycles = [corrected[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]

        # 步骤6: 单周期特征提取
        cycle_features_list = [self._extract_cycle_features(c, sampling_rate) for c in cycles]
        cycle_features_list = [f for f in cycle_features_list if f is not None]

        if not cycle_features_list:
            logging.error(f"文件 {os.path.basename(file_path)} 未能提取任何有效周期的特征")
            return None
            
        # 步骤7: 多周期统计汇总
        df_features = pd.DataFrame(cycle_features_list)
        summary_stats = {
            'peak_mean': df_features['peak'].mean(),
            'peak_std': df_features['peak'].std(),
            'peak_cv': df_features['peak'].std() / df_features['peak'].mean() if df_features['peak'].mean() > 1e-6 else 0,
            'peak_to_peak_mean': df_features['peak_to_peak'].mean(),
            'rise_slope_mean': df_features['rise_slope'].mean(),
            'symmetry_mean': df_features['symmetry'].mean()
        }

        # 额外：为域III-C计算信噪比
        signal_power = summary_stats['peak_to_peak_mean']
        # 在信号的谷值附近（低于20%峰值）计算噪声
        noise_mask = corrected < (summary_stats['peak_mean'] * 0.2)
        noise_power = np.std(corrected[noise_mask]) if len(corrected[noise_mask]) > 10 else 1e-6
        summary_stats['snr_db'] = 20 * np.log10(signal_power / noise_power) if noise_power > 0 else 50
        
        # 准备可视化数据 (降采样)
        step = max(1, len(raw_signal) // 800)
        vis_data = {
            'raw': raw_signal[::step].tolist(),
            'processed': corrected[::step].tolist(),
            'labels': time_vals[::step].tolist()
        }
        
        return {'summary': summary_stats, 'vis_data': vis_data}

    def _process_batch(self, file_map: Dict[str, str]) -> Dict[str, Any]:
        """批量处理文件，提取所需特征"""
        all_features = {}
        for key, path in file_map.items():
            result = self._process_single_file(path)
            if result:
                all_features[key] = result
        return all_features

    def calculate_qtbfs_score(self, input_data: Dict) -> Dict:
        """主入口：计算QTBFS康复评分"""
        try:
            state0_files = input_data.get('state0', {})
            current_files = input_data.get('current_state', {})

            if not current_files:
                return self._error_result("待评估状态的数据文件缺失")
            if not state0_files:
                 logging.warning("状态0（参照组）数据缺失，部分评分可能不准确")

            state0_feats = self._process_batch(state0_files)
            current_feats = self._process_batch(current_files)
            
            if not current_feats:
                return self._error_result("无法从待评估数据中提取有效特征")
            
            d1_res = self._calc_domain_I(current_feats, state0_feats)
            d2_res = self._calc_domain_II(current_feats, state0_feats)
            d3_res = self._calc_domain_III(current_feats, state0_feats)
            
            total = d1_res['total'] + d2_res['total'] + d3_res['total']
            stage, stage_code = self._determine_stage(total)
            
            vis_data = {
                'state0': {k: v['vis_data'] for k, v in state0_feats.items() if 'vis_data' in v},
                'current': {k: v['vis_data'] for k, v in current_feats.items() if 'vis_data' in v}
            }
            
            return {
                'domain_I': d1_res,
                'domain_II': d2_res,
                'domain_III': d3_res,
                'total_score': round(total, 1),
                'stage': stage,
                'stage_code': stage_code,
                'visualizations': vis_data
            }
        except Exception as e:
            logging.error(f"QTBFS评分计算顶层出错: {e}", exc_info=True)
            return self._error_result(f"计算出错: {str(e)}")

    def _calc_domain_I(self, curr: Dict, ref: Dict) -> Dict:
        # 子维度A: 峰值响应能力
        angles = ['angle_30', 'angle_45', 'angle_60', 'angle_90']
        retentions = {a: curr[a]['summary']['peak_mean'] / ref[a]['summary']['peak_mean']
                      for a in angles if a in curr and a in ref and ref[a]['summary']['peak_mean'] > 1e-6}
        avg_retention = np.mean(list(retentions.values())) if retentions else 0
        subA_score = self._get_score(avg_retention, 'domain_I_A')
        
        # 子维度B: 角度响应曲线
        common_angles = sorted([k for k in curr if k.startswith('angle_') and '120' not in k and k in ref])
        if len(common_angles) >= 5: # 至少需要5个点计算相关性
            curr_vec = [curr[k]['summary']['peak_mean'] for k in common_angles]
            ref_vec = [ref[k]['summary']['peak_mean'] for k in common_angles]
            corr = np.corrcoef(curr_vec, ref_vec)[0, 1] if np.std(curr_vec) > 0 and np.std(ref_vec) > 0 else 0
        else:
            corr = 0
        subB_score = self._get_score(corr, 'domain_I_B')

        # 子维度C: 大角度响应
        ret_120 = curr['angle_120']['summary']['peak_mean'] / ref['angle_120']['summary']['peak_mean'] \
            if 'angle_120' in curr and 'angle_120' in ref and ref['angle_120']['summary']['peak_mean'] > 1e-6 else 0
        subC_score = self._get_score(ret_120, 'domain_I_C')

        return {
            'total': subA_score + subB_score + subC_score,
            'subA_score': subA_score, 'subA_detail': {**retentions, 'avg_retention': avg_retention},
            'subB_score': subB_score, 'subB_detail': {'correlation_r': corr},
            'subC_score': subC_score, 'subC_detail': {'retention_120': ret_120}
        }

    def _calc_domain_II(self, curr: Dict, ref: Dict) -> Dict:
        # 子维度A: 速度梯度响应
        consistencies = {}
        for deg in ['30', '60', '90']:
            k1s, k3s = f"speed_{deg}_1s", f"speed_{deg}_3s"
            if k1s in curr and k3s in curr and curr[k3s]['summary']['peak_mean'] > 1e-6:
                consistencies[f'consistency_{deg}'] = 1 - abs(curr[k1s]['summary']['peak_mean'] - curr[k3s]['summary']['peak_mean']) / curr[k3s]['summary']['peak_mean']
        avg_consistency = np.mean(list(consistencies.values())) if consistencies else 0
        subA_score = self._get_score(avg_consistency, 'domain_II_A')

        # 子维度B: 重复稳定性
        cvs = [v['summary']['peak_cv'] for k, v in curr.items() if k.startswith('speed_')]
        avg_cv = np.mean(cvs) if cvs else 0.5 # 默认一个较差的值
        subB_score = self._get_score(avg_cv, 'domain_II_B')

        # 子维度C: 速度-角度交互
        speed_retentions = {k: curr[k]['summary']['peak_mean'] / ref[k]['summary']['peak_mean']
                           for k in curr if k.startswith('speed_') and k in ref and ref[k]['summary']['peak_mean'] > 1e-6}
        avg_speed_retention = np.mean(list(speed_retentions.values())) if speed_retentions else 0
        subC_score = min(8, avg_speed_retention * 8)

        return {
            'total': subA_score + subB_score + subC_score,
            'subA_score': subA_score, 'subA_detail': {**consistencies, 'avg_consistency': avg_consistency},
            'subB_score': subB_score, 'subB_detail': {'avg_cv': avg_cv},
            'subC_score': subC_score, 'subC_detail': {'avg_retention': avg_speed_retention}
        }

    def _calc_domain_III(self, curr: Dict, ref: Dict) -> Dict:
        # 子维度A: 快速响应效率
        ratios = {}
        for deg in ['30', '60', '90']:
            k1s, k3s = f"speed_{deg}_1s", f"speed_{deg}_3s"
            if k1s in curr and k3s in curr and curr[k3s]['summary']['peak_mean'] > 1e-6:
                ratios[f'ratio_{deg}'] = curr[k1s]['summary']['peak_mean'] / curr[k3s]['summary']['peak_mean']
        avg_ratio = np.mean(list(ratios.values())) if ratios else 0
        subA_score = self._get_score(avg_ratio, 'domain_III_A')
        
        # 子维度B: 极限条件响应
        k_90_1s, k_120 = 'speed_90_1s', 'angle_120'
        curr_extreme_peak = np.mean([curr[k]['summary']['peak_mean'] for k in [k_90_1s, k_120] if k in curr])
        ref_extreme_peak = np.mean([ref[k]['summary']['peak_mean'] for k in [k_90_1s, k_120] if k in ref])
        extreme_retention = curr_extreme_peak / ref_extreme_peak if ref_extreme_peak > 1e-6 else 0
        subB_score = self._get_score(extreme_retention, 'domain_III_B')

        # 子维度C: 信号质量指数
        snr_vals = [v['summary']['snr_db'] for k, v in curr.items() if 'snr_db' in v['summary']]
        avg_snr = np.mean(snr_vals) if snr_vals else 0
        subC_score = self._get_score(avg_snr, 'domain_III_C')

        return {
            'total': subA_score + subB_score + subC_score,
            'subA_score': subA_score, 'subA_detail': {**ratios, 'avg_ratio': avg_ratio},
            'subB_score': subB_score, 'subB_detail': {'extreme_retention': extreme_retention},
            'subC_score': subC_score, 'subC_detail': {'snr_db': avg_snr}
        }

    def _determine_stage(self, total: float) -> Tuple[str, int]:
        if total >= 85: return "完全康复/正常", 0
        if total >= 70: return "康复晚期", 1
        if total >= 50: return "康复中期", 2
        if total >= 30: return "康复早期", 3
        return "严重损伤", 4

    def _error_result(self, msg: str) -> Dict:
        logging.error(f"QTBFS 返回错误: {msg}")
        return {
            'total_score': 0, 'stage': msg, 'stage_code': -1,
            'domain_I': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}},
            'domain_II': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}},
            'domain_III': {'total':0, 'subA_score':0, 'subB_score':0, 'subC_score':0, 'subA_detail':{}, 'subB_detail':{}, 'subC_detail':{}},
            'visualizations': {}
        }