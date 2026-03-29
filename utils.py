"""
utils.py  V2 - 数据读取 / 预处理 / 特征提取
兼容 CSV + Excel；自动识别 时间列 & R列
含基线漂移校正、静止段去除
"""

import numpy as np, pandas as pd, os, re
from scipy.signal import find_peaks, savgol_filter, detrend
from scipy.stats import pearsonr
from scipy.ndimage import median_filter


# ========== 1. 智能读取 ==========

def smart_read_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == '.csv':
        for enc in ('utf-8-sig','utf-8','gbk','gb2312','latin1'):
            try:
                df = pd.read_csv(fp, encoding=enc)
                if len(df.columns)>=2 and len(df)>10: return df
            except: continue
        raise ValueError(f"CSV 读取失败: {fp}")
    elif ext in ('.xlsx','.xls'):
        try: df = pd.read_excel(fp, engine='openpyxl')
        except:
            try: df = pd.read_excel(fp, engine='xlrd')
            except: raise ValueError(f"Excel 读取失败: {fp}")
        if len(df.columns)>=2 and len(df)>10: return df
        raise ValueError(f"数据不足: {fp}")
    raise ValueError(f"不支持格式: {ext}")


def find_columns(df):
    """自动匹配 时间列 & 电阻 R 列 & ΔR/R 列"""
    cols = df.columns.tolist()
    time_col = r_col = dr_col = None

    # ── 时间列 ──
    for c in cols:
        s = str(c).strip().lower()
        if any(k in s for k in ('相对时间','relative','time(s)','time (s)')):
            time_col = c; break
    if not time_col:
        for c in cols:
            s = str(c).strip().lower()
            if '时间' in s or 'time' in s:
                time_col = c; break

    # ── ΔR/R 列 (新增，优先识别) ──
    for c in cols:
        s = str(c).strip()
        # 匹配 ΔR/R, △R/R, delta R/R, DR/R 等变体
        if re.match(r'^[Δ△]?\s*R\s*/\s*R(?:\s*[₀0])?\s*$', s, re.I) or \
           re.match(r'^[Δ△]?\s*R\s*/\s*R\s*$', s, re.I) or \
           'delta r/r' in s.lower() or \
           'dr/r' in s.lower():
            dr_col = c; break
    
    # ── R 列 ──
    if not dr_col:  # 只有没有 ΔR/R 列时才找 R 列
        for c in cols:
            s = str(c).strip()
            if re.match(r'^R\s*[（(].*[Ω)）]', s, re.I) or s.strip().upper()=='R':
                r_col = c; break
        if not r_col:
            for c in cols:
                s = str(c).lower()
                if 'ω' in s or 'ohm' in s or '电阻' in s:
                    r_col = c; break

    # ── 兜底 ──
    if time_col is None or (r_col is None and dr_col is None):
        num = [c for c in cols
               if str(c).strip().lower() not in ('index','序号','no','no.','#')
               and pd.to_numeric(df[c], errors='coerce').notna().sum()>len(df)*0.5]
        if time_col is None and len(num)>=2: time_col = num[0]
        if r_col is None and dr_col is None and len(num)>=2: 
            r_col = num[1]

    if time_col is None: raise ValueError(f"找不到时间列，列名={cols}")
    if r_col is None and dr_col is None: 
        raise ValueError(f"找不到电阻列或 ΔR/R 列，列名={cols}")
    return time_col, r_col, dr_col


def load_and_normalize(fp):
    df = smart_read_file(fp)
    tc, rc, drc = find_columns(df)
    
    # 读取时间列
    t = pd.to_numeric(df[tc], errors='coerce').values.astype(float)
    
    # 优先使用 ΔR/R 列，如果没有则从 R 列计算
    if drc is not None:
        # 直接使用已有的 ΔR/R 列（已经是百分比形式，如 5.2 表示 5.2%）
        dr_percent = pd.to_numeric(df[drc], errors='coerce').values.astype(float)
        
        # 转换为小数形式（除以 100），与原公式保持一致
        # 例如：5.2% → 0.052
        dr = dr_percent / 100.0
        
        m = ~(np.isnan(t)|np.isnan(dr)); t=t[m]; dr=dr[m]
        if len(t)<20: raise ValueError("有效数据 <20 行")
        
        info = dict(time_col=tc, r_col=None, dr_col=drc, 
                    n_points=len(t), duration_s=float(t[-1]-t[0]),
                    r_min=1.0, r_max=float(np.max(dr_percent)),
                    filename=os.path.basename(fp),
                    using_existing_dr=True)
    else:
        # 从 R 列计算 ΔR/R
        r = pd.to_numeric(df[rc], errors='coerce').values.astype(float)
        m = ~(np.isnan(t)|np.isnan(r)); t=t[m]; r=r[m]
        if len(t)<20: raise ValueError("有效数据 <20 行")
        rmin = np.min(r)
        if rmin<=0: rmin = np.percentile(r,1)
        if rmin<=0: rmin = 1.0
        dr = np.abs(r-rmin)/rmin
        info = dict(time_col=tc, r_col=rc, dr_col=None,
                    n_points=len(t), duration_s=float(t[-1]-t[0]),
                    r_min=float(rmin), r_max=float(np.max(r)),
                    filename=os.path.basename(fp),
                    using_existing_dr=False)
    
    return t, dr, info


# ========== 2. 预处理 ==========

def preprocess_signal(time, sig, correct_baseline=True, remove_quiet=True,
                      quiet_ratio=0.10):
    if correct_baseline:
        sig = detrend(sig, type='linear')
        sig = sig - np.min(sig)

    if remove_quiet and len(sig)>50:
        dt = np.median(np.diff(time))
        fs = 1.0/dt if dt>0 else 100.0
        ws = max(int(fs*0.5),5)
        rstd = np.array([np.std(sig[max(0,i-ws):i+ws]) for i in range(len(sig))])
        thr = np.std(sig)*quiet_ratio
        act = np.where(rstd>thr)[0]
        if len(act)>0:
            s = max(0, act[0]-int(fs*0.3))
            e = min(len(sig)-1, act[-1]+int(fs*0.3))
            time, sig = time[s:e+1], sig[s:e+1]

    dt = np.median(np.diff(time))
    fs = 1.0/dt if dt>0 else 100.0
    w = max(5, int(fs*0.02)); w = w|1
    if w<len(sig): sig = savgol_filter(sig, w, 3)
    return time, sig


# ========== 3. 峰值检测 ==========

def extract_peaks(time, sig, min_period_s=0.3):
    dt = np.median(np.diff(time))
    fs = 1.0/dt if dt>0 else 100.0
    md = max(1, int(fs*min_period_s))
    sr = np.max(sig)-np.min(sig)
    ht = np.min(sig)+0.15*sr
    pk, pr = find_peaks(sig, height=ht, distance=md, prominence=0.05*sr)
    if len(pk)==0:
        ht = np.min(sig)+0.05*sr
        pk, pr = find_peaks(sig, height=ht, distance=md)
    return pk, sig[pk], time[pk]


# ========== 4. 全特征提取 ==========

def compute_features(time, sig):
    pk, pv, pt = extract_peaks(time, sig)
    if len(pk)<2: return None
    f = {}
    f['n_cycles']   = int(len(pk))
    f['peak_mean']  = float(np.mean(pv))
    f['peak_std']   = float(np.std(pv))
    f['peak_max']   = float(np.max(pv))
    f['peak_min']   = float(np.min(pv))
    f['CV_percent'] = float(f['peak_std']/f['peak_mean']*100) if f['peak_mean']>0 else 999.
    f['AUC']        = float(np.trapezoid(np.abs(sig), time))
    # SNR
    dt=np.median(np.diff(time)); fs=1./dt if dt>0 else 100.
    w=max(5,int(fs*0.02)); w=w|1
    sm = savgol_filter(sig,w,3) if w<len(sig) else sig
    ns = sig-sm; sp=np.mean(sm**2); np_=np.mean(ns**2)
    f['SNR_dB'] = float(10*np.log10(sp/np_)) if np_>1e-15 else 60.
    # 斜率
    vl,_,_ = extract_peaks(time,-sig,0.2)
    rs=[]; fl=[]
    for p in pk:
        pv_ = vl[vl<p]
        if len(pv_)>0:
            vi=pv_[-1]; d=time[p]-time[vi]
            if d>0: rs.append(float((sig[p]-sig[vi])/d))
        nv_ = vl[vl>p]
        if len(nv_)>0:
            vi=nv_[0]; d=time[vi]-time[p]
            if d>0: fl.append(float((sig[p]-sig[vi])/d))
    f['rise_slope'] = float(np.mean(rs)) if rs else 0.
    f['fall_slope'] = float(np.mean(fl)) if fl else 0.
    f['symmetry']   = float(f['rise_slope']/f['fall_slope']) if f['fall_slope']>1e-10 else 0.
    if len(pt)>=2:
        per=np.diff(pt); f['period_mean']=float(np.mean(per)); f['period_std']=float(np.std(per))
    else:
        f['period_mean']=0.; f['period_std']=0.
    return f