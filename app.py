"""
app.py  V2 — Flask 网页应用
新增：波形预览→手动选时间段→再出图
     全局绘图参数面板
     四边框 + 刻度朝外 + 顶刊配色
"""

import os, json, uuid, traceback
import numpy as np, pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import pearsonr
from flask import (Flask, render_template, request,
                   send_from_directory, jsonify)
from werkzeug.utils import secure_filename

from utils import (load_and_normalize, preprocess_signal,
                   extract_peaks, compute_features)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024

UPLOAD  = os.path.join(os.path.dirname(__file__), 'uploads')
FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(UPLOAD,  exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

ALLOWED = {'csv','xlsx','xls'}

# ── 顶刊配色 (Nature / Science 常用) ──
PALETTE = {
    'blue':    '#0C5DA5',
    'red':     '#FF2C00',
    'green':   '#00B945',
    'orange':  '#FF9500',
    'purple':  '#845B97',
    'cyan':    '#474747',
    'yellow':  '#F0E442',
    'gray':    '#9E9E9E',
}
STATE_COLORS = {
    0:'#0C5DA5', 1:'#00B945', 2:'#F0E442',
    3:'#FF9500', 4:'#FF2C00'
}
STATE_LABELS = {
    0:'Healthy (S0)', 1:'Mild (S1)', 2:'Moderate (S2)',
    3:'Severe (S3)', 4:'Critical (S4)'
}

# ── 默认绘图参数 ──
DEFAULT_STYLE = {
    'font_family':     'Arial',
    'title_size':      48,
    'axis_label_size': 48,
    'tick_label_size': 36,
    'legend_size':     32,
    'line_width':      10.0,
    'frame_width':     4.0,
    'tick_width':      2.0,
    'tick_length':     8,
    'minor_tick_length': 4,
    'dpi':             600,
}


def get_style(form=None):
    """从请求表单读取用户自定义样式，缺省用默认值"""
    s = dict(DEFAULT_STYLE)
    if form:
        for k in s:
            v = form.get(f'style_{k}')
            if v:
                try: s[k] = float(v)
                except: pass
    return s


def apply_style(fig, axes_list, sty):
    """统一应用样式到所有 axes"""
    # 设置全局字体为 Arial，并配置 SVG 文字可编辑
    plt.rcParams['font.family'] = sty.get('font_family', 'Arial')
    plt.rcParams['font.sans-serif'] = [sty.get('font_family', 'Arial')]
    # 设置 SVG 字体类型为 none（文字保持为文本对象，可在 AI 中直接编辑内容）
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 'truetype'
    # 启用 mathtext，使用 LaTeX 风格但不依赖外部 LaTeX
    plt.rcParams['mathtext.fontset'] = 'custom'  # 使用自定义字体
    plt.rcParams['mathtext.rm'] = 'Arial'  # 罗马字体使用 Arial
    plt.rcParams['mathtext.it'] = 'Arial:italic'  # 斜体使用 Arial italic
    plt.rcParams['mathtext.bf'] = 'Arial:bold'  # 粗体使用 Arial bold
    
    for ax in axes_list:
        # 四边框全部显示 + 加粗
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(sty['frame_width'])

        # 刻度朝外 + 加粗
        ax.tick_params(axis='both', which='major',
                       direction='out',
                       width=sty['tick_width'],
                       length=sty['tick_length'],
                       labelsize=sty['tick_label_size'],
                       top=False, right=False)
        ax.tick_params(axis='both', which='minor',
                       direction='out',
                       width=sty['tick_width']*0.7,
                       length=sty['minor_tick_length'],
                       top=False, right=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # 横坐标数字加粗
        ax.tick_params(axis='x', which='major', labelsize=sty['tick_label_size'])
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        
        # 纵坐标数字加粗
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # 标签字号 - 全部加粗
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(),
                          fontsize=sty['axis_label_size'], fontweight='bold')
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(),
                          fontsize=sty['axis_label_size'], fontweight='bold')
        if ax.get_title():
            ax.set_title(ax.get_title(),
                         fontsize=sty['title_size'], fontweight='bold')
        
        # 图例文字加粗，去除外框
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_fontsize(sty['legend_size'])
                t.set_fontweight('bold')
            # 去除图例外框
            leg.get_frame().set_visible(False)
            leg.get_frame().set_linewidth(0)


def save_fig(fig, sty, prefix='fig'):
    name = f'{prefix}_{uuid.uuid4().hex[:8]}'
    png = os.path.join(FIGURES, name+'.png')
    svg = os.path.join(FIGURES, name+'.svg')
    fig.savefig(png, dpi=sty['dpi'], bbox_inches='tight', facecolor='white')
    fig.savefig(svg, dpi=sty['dpi'], bbox_inches='tight', facecolor='white', format='svg')
    plt.close(fig)
    return name+'.png', name+'.svg'


def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED


def save_upload(f, sub=''):
    if not f or not f.filename or not allowed_file(f.filename):
        return None
    fn = secure_filename(f.filename)
    if len(fn)<5: fn = uuid.uuid4().hex[:8]+'_'+f.filename
    d = os.path.join(UPLOAD, sub); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fn); f.save(p); return p


def process_file(fp, t_start=None, t_end=None):
    """读取 → 归一化 → 预处理 → 可选截取时间段"""
    time, dr, info = load_and_normalize(fp)
    time, sig = preprocess_signal(time, dr)
    # 用户手动选了时间段
    if t_start is not None or t_end is not None:
        ts = float(t_start) if t_start else time[0]
        te = float(t_end)   if t_end   else time[-1]
        mask = (time>=ts)&(time<=te)
        time, sig = time[mask], sig[mask]
        if len(time)<20:
            raise ValueError(f"选定时间段 [{ts:.2f}, {te:.2f}]s 内有效点不足")
    feat = compute_features(time, sig)
    return time, sig, info, feat


# ================= 路由 =================

@app.route('/')
def index():
    return render_template('index.html')


# ── 波形预览（上传后看处理过的波形，再决定截取） ──
@app.route('/plot/preview', methods=['POST'])
def plot_preview():
    try:
        f = request.files.get('datafile')
        if not f: return jsonify(error='请上传文件'),400
        fp = save_upload(f,'preview')
        if not fp: return jsonify(error='格式不支持'),400
        sty = get_style(request.form)

        time, dr, info = load_and_normalize(fp)
        time_p, sig_p = preprocess_signal(time, dr)
        pk, pv, pt = extract_peaks(time_p, sig_p)

        # ---- 绘制可交互预览 ----
        fig, axes = plt.subplots(2, 1, figsize=(18, 14), height_ratios=[1, 1])

        ax = axes[0]
        ax.plot(time-time[0], dr, color=PALETTE['gray'],
                lw=0.5, alpha=0.6, label='Raw (normalized)')
        ax.set_xlabel('Time (s)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$ (%)', labelpad=25)
        ax.set_title('(a) Raw Signal (before processing)', pad=30)
        ax.legend(loc='upper left', frameon=False)
        setup_left_axis_only(ax, sty)

        ax = axes[1]
        tr = time_p - time_p[0]
        ax.plot(tr, sig_p, color=PALETTE['blue'], lw=sty['line_width']*0.3, label='Processed')
        ax.plot(pt-time_p[0], pv, 'v', color=PALETTE['red'],
                ms=8, label=f'Peaks: {len(pk)}')
        ax.set_xlabel('Time (s)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$ (%)', labelpad=25)
        ax.set_title(f'(b) Processed Signal  |  Duration: {tr[-1]:.2f}s  |  '
                     f'Peaks: {len(pk)}', pad=30)
        ax.legend(loc='upper left', frameon=False)
        setup_left_axis_only(ax, sty)
        
        # 调整横坐标刻度，避免黏连
        for ax in axes:
            ax.tick_params(axis='x', rotation=0)
            # 自动调整刻度位置
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

        apply_style(fig, list(fig.axes), sty)
        plt.tight_layout(pad=5.0)
        png, svg = save_fig(fig, sty, 'preview')

        res = dict(figure=png, svg=svg,
                   time_min=round(float(time_p[0]),3),
                   time_max=round(float(time_p[-1]),3),
                   info={k:(round(v,3) if isinstance(v,float) else v)
                         for k,v in info.items()})
        if pk is not None and len(pk)>0:
            fe = compute_features(time_p, sig_p)
            if fe: res['features'] = {k:round(v,4) if isinstance(v,float) else v
                                       for k,v in fe.items()}
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)),500


# ── 标定曲线 ──
@app.route('/plot/calibration', methods=['POST'])
def plot_calibration():
    try:
        files = request.files.getlist('datafiles')
        angles = [float(a) for a in request.form.get('angles','').split(',') if a.strip()]
        t_starts = request.form.get('t_starts','').split(',')
        t_ends   = request.form.get('t_ends','').split(',')
        sty = get_style(request.form)

        if len(angles)!=len(files):
            return jsonify(error=f'角度数({len(angles)})≠文件数({len(files)})'),400

        angle_peaks = {}
        for i,(f,ang) in enumerate(zip(files,angles)):
            fp = save_upload(f,'calib')
            if not fp: continue
            ts = t_starts[i].strip() if i<len(t_starts) and t_starts[i].strip() else None
            te = t_ends[i].strip()   if i<len(t_ends)   and t_ends[i].strip()   else None
            t,s,_,_ = process_file(fp, ts, te)
            _,pv,_ = extract_peaks(t,s)
            if len(pv)>0: angle_peaks[ang] = pv.tolist()

        if len(angle_peaks)<2:
            return jsonify(error='有效角度<2'),400

        sa = sorted(angle_peaks.keys())
        mn = [np.mean(angle_peaks[a]) for a in sa]
        sd = [np.std(angle_peaks[a])  for a in sa]

        fig, axes = plt.subplots(1,3, figsize=(24, 9))

        # (a) 标定
        ax = axes[0]
        ax.errorbar(sa, mn, yerr=sd, fmt='o-', color=PALETTE['blue'],
                    capsize=6, ms=10, lw=sty['line_width'], label='Measured')
        c = np.polyfit(sa,mn,1)
        xf = np.linspace(min(sa),max(sa),100)
        fit = np.polyval(c,sa)
        ss_r = np.sum((np.array(mn)-fit)**2)
        ss_t = np.sum((np.array(mn)-np.mean(mn))**2)
        r2 = 1-ss_r/ss_t if ss_t>0 else 0
        ax.plot(xf, np.polyval(c,xf), '--', color=PALETTE['red'],
                lw=sty['line_width'], label=f'Linear fit (R²={r2:.4f})')
        ax.text(0.05,0.88, f'GF = {c[0]:.5f} /°', transform=ax.transAxes,
                fontsize=sty['tick_label_size'], fontweight='bold',
                bbox=dict(boxstyle='round',fc='#FFF9C4',alpha=0.9))
        ax.set_xlabel('Bending Angle (°)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.legend(loc='best', frameon=False)
        ax.set_title('(a) Calibration Curve', pad=30)
        setup_left_axis_only(ax, sty)

        # (b) 箱线
        ax = axes[1]
        bp = ax.boxplot([angle_peaks[a] for a in sa],
                        positions=range(len(sa)), widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=PALETTE['blue'], alpha=0.4),
                        medianprops=dict(color=PALETTE['red'], lw=3))
        ax.set_xticks(range(len(sa)))
        ax.set_xticklabels([f'{int(a)}°' for a in sa], rotation=45)
        ax.set_xlabel('Angle', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.set_title('(b) Peak Distribution', pad=30)
        setup_left_axis_only(ax, sty)

        # (c) 局部灵敏度
        ax = axes[2]
        if len(sa)>=3:
            da = np.diff(sa); dp = np.diff(mn)
            sens = dp/da
            mid = [(sa[i]+sa[i+1])/2 for i in range(len(sa)-1)]
            ax.bar(range(len(mid)), sens, color=PALETTE['orange'],
                   edgecolor='white', width=0.6)
            ax.set_xticks(range(len(mid)))
            ax.set_xticklabels([f'{int(m)}°' for m in mid], rotation=45)
        ax.set_ylabel('Sensitivity (/°)', labelpad=20)
        ax.set_title('(c) Local Sensitivity', pad=30)
        setup_left_axis_only(ax, sty)

        apply_style(fig,list(fig.axes),sty)
        plt.tight_layout(pad=3.0)
        png,svg = save_fig(fig,sty,'calibration')
        return jsonify(figure=png, svg=svg,
                       calibration=dict(R2=round(r2,5), GF=round(c[0],6)))
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


# ── 速度对比 ──
@app.route('/plot/speed_comparison', methods=['POST'])
def plot_speed_comparison():
    try:
        sty = get_style(request.form)
        angle = request.form.get('angle','60')
        sd = {}
        for label in ('1s','2s','3s'):
            f = request.files.get(f'file_{label}')
            if f and f.filename:
                fp = save_upload(f,'speed')
                ts = request.form.get(f't_start_{label}') or None
                te = request.form.get(f't_end_{label}')   or None
                t,s,_,fe = process_file(fp,ts,te)
                sd[label] = (t,s,fe)
        if len(sd)<2: return jsonify(error='至少上传2个文件'),400

        sc = {'1s':PALETTE['red'],'2s':PALETTE['orange'],'3s':PALETTE['blue']}
        sn = {'1s':'Fast (1 s/cyc)','2s':'Medium (2 s/cyc)','3s':'Slow (3 s/cyc)'}

        fig, axes = plt.subplots(1,3,figsize=(24, 9))

        # (a) 波形叠加
        ax = axes[0]
        for sp in ('1s','2s','3s'):
            if sp not in sd: continue
            t,s,_ = sd[sp]; tr=t-t[0]
            mt = min(15,tr[-1]); m=tr<=mt
            ax.plot(tr[m],s[m], color=sc[sp], lw=sty['line_width']*0.4,
                    label=sn[sp], alpha=0.85)
        ax.set_xlabel('Time (s)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.legend(loc='upper right', frameon=False)
        ax.set_title(f'(a) Speed Comparison at {angle}°', pad=30)
        setup_left_axis_only(ax, sty)

        # (b) 峰值柱状图
        ax = axes[1]
        sps = [s for s in ('1s','2s','3s') if s in sd]
        pm,ps_ = [],[]
        for sp in sps:
            t,s,_ = sd[sp]; _,pv,_ = extract_peaks(t,s)
            pm.append(np.mean(pv) if len(pv) else 0)
            ps_.append(np.std(pv) if len(pv) else 0)
        x = np.arange(len(sps))
        bars = ax.bar(x, pm, yerr=ps_, color=[sc[s] for s in sps],
                      edgecolor='white', width=0.5, capsize=6)
        ax.set_xticks(x)
        # 减小字号并旋转，避免重叠
        ax.set_xticklabels([sn[s] for s in sps], fontsize=20, rotation=15, ha='right')
        ax.set_xlabel('Speed', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.set_title('(b) Peak Amplitude', pad=30)
        setup_left_axis_only(ax, sty)
        # 放大数字到 28
        for b,v in zip(bars,pm):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{v:.4f}', ha='center', fontsize=28)

        # (c) 一致性
        ax = axes[2]
        if '1s' in sd and '3s' in sd:
            t1,s1,_=sd['1s']; t3,s3,_=sd['3s']
            _,pv1,_=extract_peaks(t1,s1); _,pv3,_=extract_peaks(t3,s3)
            m1=np.mean(pv1) if len(pv1) else 0
            m3=np.mean(pv3) if len(pv3) else 0
            con = (1-abs(m1-m3)/m3)*100 if m3>0 else 0
            rat = m1/m3*100 if m3>0 else 0
            ax.bar(['Consistency','Fast/Slow\nRatio'],[max(0,con),rat],
                   color=[PALETTE['green'],PALETTE['blue']],
                   edgecolor='white', width=0.5)
            ax.axhline(90,color=PALETTE['gray'],ls=':',lw=2)
        ax.set_ylabel('%', labelpad=20)
        ax.set_title('(c) Consistency Index', pad=30)
        setup_left_axis_only(ax, sty)

        apply_style(fig,list(fig.axes),sty)
        plt.tight_layout(pad=3.0)
        png,pdf = save_fig(fig,sty,'speed')
        return jsonify(figure=png, svg=pdf)
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


# ── 疲劳 ──
@app.route('/plot/fatigue', methods=['POST'])
def plot_fatigue():
    try:
        f = request.files.get('datafile')
        if not f: return jsonify(error='请上传文件'),400
        fp = save_upload(f,'fatigue')
        sty = get_style(request.form)
        ts = request.form.get('t_start') or None
        te = request.form.get('t_end')   or None
        time,sig,info,_ = process_file(fp,ts,te)
        pk,pv,pt = extract_peaks(time,sig)
        if len(pv)<5: return jsonify(error='周期<5'),400

        cy = np.arange(1,len(pv)+1)
        mv = np.mean(pv); sv = np.std(pv)
        cv = sv/mv*100 if mv>0 else 0

        fig, axes = plt.subplots(1,3,figsize=(17,5.5))

        # (a)
        ax = axes[0]
        tr = time-time[0]
        ax.fill_between(tr,sig,alpha=0.25,color=PALETTE['blue'])
        ax.plot(tr,sig,color=PALETTE['blue'],lw=0.5)
        ax.set_xlabel('Time (s)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$ (%)', labelpad=25)
        ax.set_title(f'(a) Full Waveform ({len(pv)} cyc)', pad=30)
        setup_left_axis_only(ax, sty)

        # (b)
        ax = axes[1]
        ax.plot(cy,pv,'o-',color=PALETTE['blue'],ms=6,lw=sty['line_width']*0.4)
        ax.axhline(mv,color=PALETTE['red'],ls='--',lw=2,label=f'Mean={mv:.4f}')
        ax.fill_between(cy,mv-sv,mv+sv,alpha=0.12,color=PALETTE['red'],label='±1 SD')
        ax.set_xlabel('Cycle #', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.legend(loc='upper right', frameon=False)
        ax.text(0.95,0.06,f'CV = {cv:.2f}%',transform=ax.transAxes,ha='right',
                fontsize=sty['axis_label_size'],fontweight='bold',
                bbox=dict(boxstyle='round',fc='#FFF9C4'))
        ax.set_title('(b) Peak Stability', pad=30)
        setup_left_axis_only(ax, sty)

        # (c) 前后对比
        ax = axes[2]
        n=len(pv); n10=max(3,n//10)
        f10=pv[:n10]; l10=pv[-n10:]
        ret = np.mean(l10)/np.mean(f10)*100 if np.mean(f10)>0 else 0
        ax.bar(['First 10%','Last 10%'],[np.mean(f10),np.mean(l10)],
               yerr=[np.std(f10),np.std(l10)],
               color=[PALETTE['blue'],PALETTE['orange']],
               edgecolor='white',width=0.5,capsize=6)
        ax.set_xlabel('Period', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        ax.text(0.5,0.06,f'Retention: {ret:.1f}%',transform=ax.transAxes,
                ha='center',fontsize=sty['tick_label_size'],fontweight='bold',
                bbox=dict(boxstyle='round',fc='#FFF9C4'))
        ax.set_title('(c) Fatigue Retention', pad=30)
        setup_left_axis_only(ax, sty)

        apply_style(fig,list(fig.axes),sty)
        plt.tight_layout()
        png,pdf = save_fig(fig,sty,'fatigue')
        return jsonify(figure=png, svg=pdf,
                       stats=dict(cycles=int(len(pv)), mean=round(mv,5),
                                  CV=round(cv,2), retention=round(ret,1)))
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


# ── 多State ──
@app.route('/plot/multi_state', methods=['POST'])
def plot_multi_state():
    try:
        files  = request.files.getlist('datafiles')
        states = [int(s) for s in request.form.get('states','').split(',') if s.strip()]
        sty = get_style(request.form)
        if len(files)!=len(states):
            return jsonify(error='文件数≠State数'),400

        sdata = {}
        for f,st in zip(files,states):
            fp = save_upload(f,'ms')
            if not fp: continue
            t,s,info,fe = process_file(fp)
            _,pv,_ = extract_peaks(t,s)
            sdata[st] = dict(t=t,s=s,pm=np.mean(pv) if len(pv) else 0,
                             pv=pv, fn=info['filename'])

        if len(sdata)<2: return jsonify(error='有效State<2'),400

        fig, axes = plt.subplots(1,3,figsize=(24, 9))
        ss = sorted(sdata.keys())

        # (a)
        ax = axes[0]
        for st in ss:
            d=sdata[st]; tr=d['t']-d['t'][0]; mt=min(10,tr[-1]); m=tr<=mt
            ax.plot(tr[m],d['s'][m],color=STATE_COLORS.get(st,'#888'),
                    lw=sty['line_width']*0.4,alpha=0.85,
                    label=STATE_LABELS.get(st,f'S{st}'))
        ax.set_xlabel('Time (s)', labelpad=20)
        ax.set_ylabel(r'$\mathbf{\Delta}$R/R$_{\mathbf{0}}$', labelpad=25)
        # 图例放在左上角，避免与曲线重叠
        ax.legend(loc='upper left', frameon=False, fontsize=24)
        ax.set_title('(a) Multi-State Waveforms', pad=30)
        setup_left_axis_only(ax, sty)

        # (b) 保留率
        ax = axes[1]
        base = sdata.get(0,{}).get('pm') or max(d['pm'] for d in sdata.values())
        rets = [sdata[st]['pm']/base*100 if base>0 else 0 for st in ss]
        bars = ax.bar([STATE_LABELS.get(s,f'S{s}') for s in ss], rets,
                      color=[STATE_COLORS.get(s,'#888') for s in ss],
                      edgecolor='white', width=0.55)
        ax.axhline(100,color=PALETTE['gray'],ls='--',lw=1.5)
        # 减小横坐标字号并旋转，避免重叠
        ax.set_xlabel('State', labelpad=20)
        ax.set_ylabel('Peak Retention (%)', labelpad=25)
        ax.set_title('(b) Signal Retention', pad=30)
        ax.set_xticks(range(len(ss)))
        ax.set_xticklabels([STATE_LABELS.get(s,f'S{s}') for s in ss], 
                          fontsize=18, rotation=45, ha='right')
        for b,v in zip(bars,rets):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                    f'{v:.1f}%', ha='center', fontsize=20,
                    fontweight='bold')

        # (c) QTBFS
        ax = axes[2]
        ax.axhspan(0,49,alpha=0.08,color=PALETTE['red'])
        ax.axhspan(49,84,alpha=0.08,color=PALETTE['orange'])
        ax.axhspan(84,100,alpha=0.08,color=PALETTE['green'])
        scores = [min(100,r) for r in rets]
        bars = ax.bar([STATE_LABELS.get(s,f'S{s}') for s in ss], scores,
                      color=[STATE_COLORS.get(s,'#888') for s in ss],
                      edgecolor='white', width=0.55)
        ax.set_ylim(0,110)
        # 减小横坐标字号并旋转，避免重叠
        ax.set_xlabel('State', labelpad=20)
        ax.set_ylabel('QTBFS Score', labelpad=25)
        ax.set_title('(c) Rehabilitation Score', pad=30)
        ax.set_xticks(range(len(ss)))
        ax.set_xticklabels([STATE_LABELS.get(s,f'S{s}') for s in ss], 
                          fontsize=18, rotation=45, ha='right')
        for b,v in zip(bars,scores):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                    f'{v:.1f}', ha='center', fontsize=20,
                    fontweight='bold')

        apply_style(fig,list(fig.axes),sty)
        plt.tight_layout(pad=3.0)
        png,pdf = save_fig(fig,sty,'multistate')
        return jsonify(figure=png,svg=pdf)
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


# ── 热力图 ──
@app.route('/plot/heatmap', methods=['POST'])
def plot_heatmap():
    try:
        sty = get_style(request.form)
        angles = [30,60,90]; speeds = ['1s','2s','3s']
        mat = np.zeros((3,3))
        for i,a in enumerate(angles):
            for j,sp in enumerate(speeds):
                f = request.files.get(f'file_{a}_{sp}')
                if f and f.filename:
                    fp = save_upload(f,'hm')
                    t,s,_,_ = process_file(fp)
                    _,pv,_ = extract_peaks(t,s)
                    mat[i,j] = np.mean(pv) if len(pv) else 0

        ret = np.zeros((3,3))
        for i in range(3):
            b = mat[i,2]
            for j in range(3):
                ret[i,j] = mat[i,j]/b*100 if b>0 else 0

        fig, axes = plt.subplots(1,2,figsize=(20, 9))

        ax = axes[0]
        im = ax.imshow(mat, cmap='Blues', aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Fast(1s)','Med(2s)','Slow(3s)'], fontsize=20, rotation=15, ha='right')
        ax.set_yticks(range(3)); ax.set_yticklabels(['30°','60°','90°'], fontsize=20)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',
                        fontsize=sty['tick_label_size'],fontweight='bold')
        plt.colorbar(im,ax=ax,label=r'Peak $\mathbf{\Delta}$R/R$_{\mathbf{0}}$',shrink=0.85, pad=0.02)
        ax.set_xlabel('Speed', labelpad=20)
        ax.set_ylabel('Angle', labelpad=20)
        ax.set_title('(a) Absolute Response', pad=30)

        ax = axes[1]
        im2 = ax.imshow(ret, cmap='RdYlGn', vmin=70, vmax=110, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Fast(1s)','Med(2s)','Slow(3s)'], fontsize=20, rotation=15, ha='right')
        ax.set_yticks(range(3)); ax.set_yticklabels(['30°','60°','90°'], fontsize=20)
        for i in range(3):
            for j in range(3):
                c = 'white' if ret[i,j]<85 else 'black'
                ax.text(j,i,f'{ret[i,j]:.1f}%',ha='center',va='center',
                        fontsize=sty['tick_label_size'],fontweight='bold',color=c)
        plt.colorbar(im2,ax=ax,label='Retention %',shrink=0.85, pad=0.02)
        ax.set_xlabel('Speed', labelpad=20)
        ax.set_ylabel('Angle', labelpad=20)
        ax.set_title('(b) Retention Matrix', pad=30)

        apply_style(fig,list(fig.axes),sty)
        plt.tight_layout(pad=3.0)
        png,pdf = save_fig(fig,sty,'heatmap')
        return jsonify(figure=png,svg=pdf,
                       raw=mat.tolist(), retention=ret.tolist())
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


# ── 相关性矩阵 ──
@app.route('/plot/correlation', methods=['POST'])
def plot_correlation():
    try:
        files = request.files.getlist('datafiles')
        sty = get_style(request.form)
        if len(files)<5: return jsonify(error='至少需要5个文件'),400

        recs = []
        for f in files:
            fp = save_upload(f,'corr')
            if not fp: continue
            t,s,info,fe = process_file(fp)
            if fe:
                fe['filename'] = info['filename']
                recs.append(fe)
        if len(recs)<5: return jsonify(error='有效数据<5'),400

        df = pd.DataFrame(recs)
        cols = [c for c in ('peak_mean','rise_slope','fall_slope',
                            'symmetry','CV_percent','AUC','SNR_dB') if c in df.columns]
        short = {'peak_mean':'Peak','rise_slope':'Rise','fall_slope':'Decay',
                 'symmetry':'Sym','CV_percent':'CV','AUC':'AUC','SNR_dB':'SNR'}
        pdf_ = df[cols].rename(columns=short).dropna()
        n = len(pdf_.columns)
        if n<3: return jsonify(error='特征列<3'),400

        fig, axes = plt.subplots(n,n, figsize=(4.5*n, 4.5*n))
        cmap = plt.cm.Blues
        cr = {}

        for i in range(n):
            for j in range(n):
                ax = axes[i,j]
                ci = pdf_.columns[i]; cj = pdf_.columns[j]
                if i==j:
                    ax.hist(pdf_[ci], bins=12, density=True, alpha=0.5,
                            color=PALETTE['blue'], edgecolor='white')
                    ax.set_yticks([])
                elif i<j:
                    vv = pdf_[[cj,ci]].dropna()
                    r,p = pearsonr(vv[cj],vv[ci]) if len(vv)>=3 else (0,1)
                    ax.set_facecolor(cmap(min(abs(r),1.)))
                    tc = 'white' if abs(r)>0.5 else 'black'
                    ax.text(0.5,0.55,f'{r:.2f}',transform=ax.transAxes,
                            ha='center',va='center',fontsize=18,
                            fontweight='bold',color=tc)
                    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
                    ax.text(0.5,0.25,sig,transform=ax.transAxes,ha='center',
                            fontsize=14,color=tc,alpha=0.7)
                    ax.set_xticks([]); ax.set_yticks([])
                    cr[f'{cj}_vs_{ci}'] = dict(r=round(r,4),p=round(p,6))
                else:
                    ax.scatter(pdf_[cj],pdf_[ci],s=20,alpha=0.65,
                               color=PALETTE['blue'],edgecolors='white',lw=0.5)
                    vv = pdf_[[cj,ci]].dropna()
                    if len(vv)>=2:
                        z = np.polyfit(vv[cj],vv[ci],1)
                        xl = np.linspace(vv[cj].min(),vv[cj].max(),30)
                        ax.plot(xl,np.polyval(z,xl),'--',color=PALETTE['red'],lw=1.5)

                # 四边框
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(sty['frame_width']*0.6)
                ax.tick_params(direction='out',width=sty['tick_width']*0.6,
                               length=sty['tick_length']*0.5)

                if i==n-1:
                    ax.set_xlabel(cj,fontsize=sty['axis_label_size']-2,fontweight='bold', labelpad=10)
                else: ax.set_xticklabels([])
                if j==0 and i>0:
                    ax.set_ylabel(ci,fontsize=sty['axis_label_size']-2,fontweight='bold', labelpad=10)
                elif j!=0: ax.set_yticklabels([])

        plt.suptitle('Feature Correlation Matrix',
                     fontsize=sty['title_size']+2,fontweight='bold',y=1.02)
        plt.tight_layout(pad=2.0)
        png,pdf = save_fig(fig,sty,'correlation')
        csv_n = png.replace('.png','_features.csv')
        df.to_csv(os.path.join(FIGURES,csv_n),index=False)
        return jsonify(figure=png,svg=pdf,features_csv=csv_n,
                       correlations=cr, n_samples=len(pdf_))
    except Exception as e:
        traceback.print_exc(); return jsonify(error=str(e)),500


@app.route('/figures/<fn>')
def serve(fn):
    return send_from_directory(FIGURES, fn)


def set_ylabel_formatted(ax, ylabel):
    """设置格式化的纵坐标标签，支持下标"""
    ax.set_ylabel(ylabel, fontsize=ax.yaxis.label.get_size(), fontweight='bold')


def setup_left_axis_only(ax, sty):
    """设置左轴只显示轴线和纵坐标标题，隐藏刻度和数字"""
    # 隐藏左轴刻度和刻度标签
    ax.yaxis.set_ticks_position('none')
    ax.set_yticklabels([])
    # 但保留左边框线（作为纵轴线）
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(sty['frame_width'])


if __name__=='__main__':
    print("\n"+"="*55)
    print("  Janus Sensor V2  →  http://127.0.0.1:5004")
    print("="*55+"\n")
    app.run(debug=True, port=5004)