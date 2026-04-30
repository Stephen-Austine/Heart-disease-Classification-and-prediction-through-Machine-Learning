"""
CardioScan AI — ECG Classification App
DSA4050 Deep Learning for Computer Vision

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, random, csv
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CardioScan AI", page_icon="🫀",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['AFib', 'Arrhythmia', 'MI', 'Normal']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
OUTPUTS_DIR   = Path('ecg_project/outputs')
CHECKPOINT    = Path('ecg_project/checkpoints/ecg_best_model.pth')
SPEC_DIR      = Path('ecg_project/ecg_spectrograms')

CLASS_INFO = {
    'Normal':     {'color':'#4ade80','risk':'Low',          'icon':'✅','desc':'Healthy sinus rhythm with regular electrical activity.'},
    'Arrhythmia': {'color':'#f87171','risk':'High',         'icon':'⚠️','desc':'Irregular heart rhythm — premature ventricular contractions detected.'},
    'AFib':       {'color':'#fb923c','risk':'Moderate–High','icon':'🔶','desc':'Atrial fibrillation — irregular, rapid electrical impulses in the atria.'},
    'MI':         {'color':'#f472b6','risk':'Critical',     'icon':'🚨','desc':'Myocardial infarction pattern — bundle branch block morphology detected.'},
}

SYNTH = {
    'Normal':     {'hr':70, 'noise':0.03,'pvc':False,'afib':False,'bbb':False},
    'Arrhythmia': {'hr':75, 'noise':0.05,'pvc':True, 'afib':False,'bbb':False},
    'AFib':       {'hr':110,'noise':0.08,'pvc':False,'afib':True, 'bbb':False},
    'MI':         {'hr':65, 'noise':0.04,'pvc':False,'afib':False,'bbb':True},
}

PAGES = [
    ('classify',   '🫀', 'Classify'),
    ('simulator',  '📡', 'Simulator'),
    ('analytics',  '📊', 'Analytics'),
    ('explorer',   '🖼️', 'Explorer'),
    ('compare',    '⚖️',  'Compare'),
    ('history',    '📋', 'History'),
    ('howitworks', '🎓', 'How It Works'),
    ('settings',   '⚙️',  'Settings'),
]

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [('page','classify'),('pred_history',[]),
                     ('last_pred',None),('last_probs',None),
                     ('theme','light')]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#070711;--surf:#0f0f1a;--surf2:#161625;--bdr:#1f1f35;--bdr2:#2a2a45;
      --acc:#7c6af7;--acc2:#a59cfa;--grn:#4ade80;--red:#f87171;--org:#fb923c;
      --pnk:#f472b6;--txt:#e2e2f0;--dim:#7070a0;--dimmer:#404060;}

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]
    {background:var(--bg)!important;color:var(--txt)!important;
     font-family:'Plus Jakarta Sans',sans-serif!important;}
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stHeader"],[data-testid="stToolbar"],footer{display:none!important;}
[data-testid="stAppViewBlockContainer"]{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"] > div:first-child {padding-top:0!important;}

/* ── Navbar row of buttons ── */
div[data-testid="stHorizontalBlock"]:first-of-type{
    background:rgba(7,7,17,0.95)!important;
    backdrop-filter:blur(20px)!important;
    border-bottom:1px solid var(--bdr)!important;
    position:sticky!important; top:0!important; z-index:9999!important;
    padding:0.5rem 1.5rem!important; gap:0.2rem!important;
    align-items:center!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="stColumn"]{
    flex:0 0 auto!important; width:auto!important; min-width:0!important; padding:0!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton>button{
    background:transparent!important; color:var(--dim)!important;
    border:none!important; border-radius:8px!important;
    font-family:'Plus Jakarta Sans',sans-serif!important;
    font-size:0.78rem!important; font-weight:600!important;
    padding:0.4rem 0.8rem!important; white-space:nowrap!important;
    box-shadow:none!important; transition:all 0.15s!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton>button:hover{
    background:var(--surf2)!important; color:var(--txt)!important;
    transform:none!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton[data-active="true"]>button,
div[data-testid="stHorizontalBlock"]:first-of-type .stButton>button[kind="primary"]{
    background:var(--acc)!important; color:#fff!important;
}

/* ── Page content ── */
.page-wrap{padding:1.8rem 2.5rem; max-width:1400px; margin:0 auto;}
.hero-title{font-size:2.2rem;font-weight:800;line-height:1.1;
            letter-spacing:-0.03em;margin:0 0 0.4rem;}
.hero-title .hl{color:var(--acc2);}
.hero-sub{font-family:'JetBrains Mono',monospace;font-size:0.76rem;
          color:var(--dim);margin:0 0 1.6rem;}

/* ── Cards ── */
.mcard{background:var(--surf);border:1px solid var(--bdr);border-radius:10px;
       padding:1.1rem 1.3rem;margin-bottom:0.9rem;}
.mcard .lbl{font-family:'JetBrains Mono',monospace;font-size:0.6rem;
            color:var(--dimmer);letter-spacing:0.15em;text-transform:uppercase;
            margin-bottom:0.35rem;}
.mcard .val{font-size:1.8rem;font-weight:800;color:var(--acc2);line-height:1;}

/* ── Result card ── */
.rcard{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;
       padding:1.6rem;margin-top:0.8rem;
       border-top:3px solid var(--acc);}
.rclass{font-size:2.1rem;font-weight:800;letter-spacing:-0.03em;line-height:1;
        margin-bottom:0.45rem;}
.rdesc{font-family:'JetBrains Mono',monospace;font-size:0.7rem;
       color:var(--dim);line-height:1.6;margin-bottom:1rem;}
.chip{display:inline-flex;align-items:center;gap:0.3rem;padding:0.22rem 0.65rem;
      border-radius:20px;font-family:'JetBrains Mono',monospace;
      font-size:0.62rem;font-weight:600;letter-spacing:0.04em;margin-right:0.5rem;}

/* ── Confidence bars ── */
.cbar{margin-bottom:0.75rem;}
.cbar-hdr{display:flex;justify-content:space-between;margin-bottom:0.25rem;}
.cbar-lbl,.cbar-pct{font-family:'JetBrains Mono',monospace;font-size:0.7rem;}
.cbar-track{height:5px;background:var(--bdr);border-radius:3px;overflow:hidden;}
.cbar-fill{height:5px;border-radius:3px;}

/* ── Section label ── */
.slbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--dimmer);
      letter-spacing:0.2em;text-transform:uppercase;
      border-bottom:1px solid var(--bdr);padding-bottom:0.4rem;margin-bottom:1rem;}

/* ── Alerts ── */
.alert{border-radius:6px;padding:0.75rem 1rem;font-family:'JetBrains Mono',monospace;
       font-size:0.7rem;margin:0.7rem 0;border-left-width:3px;border-left-style:solid;}
.awarn{background:rgba(251,146,60,.08);border-color:#1f1f35;
       border-left-color:var(--org);color:var(--org);}
.aerr {background:rgba(248,113,113,.08);border-color:#1f1f35;
       border-left-color:var(--red);color:var(--red);}
.aok  {background:rgba(74,222,128,.08);border-color:#1f1f35;
       border-left-color:var(--grn);color:var(--grn);}

/* ── Content buttons ── */
.stButton>button{background:var(--acc)!important;color:#fff!important;
    font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;
    font-size:0.82rem!important;border:none!important;border-radius:8px!important;
    padding:0.6rem 1.5rem!important;
    box-shadow:0 2px 12px rgba(124,106,247,0.3)!important;
    transition:all 0.2s!important;}
.stButton>button:hover{background:var(--acc2)!important;
    transform:translateY(-1px)!important;
    box-shadow:0 4px 20px rgba(124,106,247,0.4)!important;}

/* ── Misc ── */
[data-testid="stFileUploader"]{background:var(--surf)!important;
    border:1.5px dashed var(--bdr2)!important;border-radius:10px!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surf)!important;
    border-radius:8px!important;padding:0.3rem!important;
    border:1px solid var(--bdr)!important;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace!important;
    font-size:0.68rem!important;color:var(--dim)!important;
    border-radius:6px!important;padding:0.5rem 0.9rem!important;}
.stTabs [aria-selected="true"]{background:var(--acc)!important;color:#fff!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]
    {display:none!important;}
div[data-testid="stMarkdownContainer"] p{color:var(--dim)!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.76rem!important;
    line-height:1.7!important;}

/* ── Light theme overrides ── */
[data-theme="light"] {
    --bg:#f8f8fc; --surf:#ffffff; --surf2:#f0f0f8; --bdr:#e0e0f0; --bdr2:#c8c8e0;
    --txt:#1a1a2e; --dim:#5a5a7a; --dimmer:#8888aa;
}
[data-theme="light"] html,
[data-theme="light"] body,
[data-theme="light"] [data-testid="stAppViewContainer"],
[data-theme="light"] [data-testid="stMain"] {
    background: #f8f8fc !important; color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt):
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4,inplace=True),
                                  nn.Linear(m.classifier[1].in_features,4))
    loaded = Path(ckpt).exists()
    if loaded:
        m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m, loaded

def classify_pil(img, mdl):
    tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                              transforms.ToTensor(),
                              transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    with torch.no_grad():
        probs = F.softmax(mdl(tf(img.convert('RGB')).unsqueeze(0)),dim=1).squeeze().numpy()
    return CLASS_NAMES[probs.argmax()], probs

def sig2spec(ecg, fs=360, nperseg=64, noverlap=32):
    _,_,Z = signal.stft(ecg, fs=fs, nperseg=nperseg, noverlap=noverlap)
    s = 20*np.log10(np.abs(Z)+1e-10)
    return (s-s.min())/(s.max()-s.min()+1e-10)

def spec2pil(spec):
    fig,ax = plt.subplots(figsize=(2,2),dpi=112)
    ax.imshow(spec,aspect='auto',origin='lower',cmap='viridis'); ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf,format='png',bbox_inches='tight',pad_inches=0)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB')

def dkfig(w=9,h=3):
    fig,ax = plt.subplots(figsize=(w,h),facecolor='#070711')
    ax.set_facecolor('#070711'); ax.tick_params(colors='#404060',labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#1f1f35')
    return fig,ax

def dkfigs(r,c,w=10,h=4,**kw):
    fig,axes = plt.subplots(r,c,figsize=(w,h),facecolor='#070711',**kw)
    for ax in (axes.flat if hasattr(axes,'flat') else [axes]):
        ax.set_facecolor('#070711'); ax.tick_params(colors='#404060',labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#1f1f35')
    return fig,axes

def log(source, pred, probs, conf):
    st.session_state.pred_history.append({
        'time':datetime.now().strftime('%H:%M:%S'),'source':source,
        'prediction':pred,'confidence':f'{conf*100:.1f}%',
        'AFib':f'{probs[0]*100:.1f}%','Arrhythmia':f'{probs[1]*100:.1f}%',
        'MI':f'{probs[2]*100:.1f}%','Normal':f'{probs[3]*100:.1f}%',
    })

def bars_html(probs, pred):
    """Return confidence bars as a single HTML string (no separate st calls)."""
    html = ""
    for cls, p in zip(CLASS_NAMES, probs):
        ci     = CLASS_INFO[cls]
        active = cls == pred
        col    = ci['color'] if active else '#2a2a45'
        tcol   = ci['color'] if active else '#404060'
        arrow  = '▶ ' if active else '&nbsp;&nbsp;'
        html  += (
            '<div class="cbar">'
            '<div class="cbar-hdr">'
            f'<span class="cbar-lbl" style="color:{tcol};">{arrow}{cls}</span>'
            f'<span class="cbar-pct" style="color:{col};font-weight:700;">{p*100:.1f}%</span>'
            '</div>'
            '<div class="cbar-track">'
            f'<div class="cbar-fill" style="width:{p*100:.1f}%;background:{col};"></div>'
            '</div></div>'
        )
    return html

def gen_ecg(tname, dur=2, fs=360):
    cfg = SYNTH[tname]
    n   = int(dur*fs); ecg = np.zeros(n)
    for bt in np.arange(0, dur, 60.0/cfg['hr']):
        idx = int(bt*fs)
        pw  = int(0.08*fs); pi = max(0, idx-int(0.16*fs))
        ecg[pi:pi+pw] += 0.15*np.hanning(pw)
        qw = int(0.10*fs) if not cfg['bbb'] else int(0.16*fs)
        if idx+qw < n:
            q = np.zeros(qw); q[:qw//3]=-0.1
            q[qw//3:2*qw//3] = 1.2 if not cfg['pvc'] else -1.0
            q[2*qw//3:] = -0.3 if not cfg['bbb'] else 0.3
            ecg[idx:idx+qw] += q
        ti=idx+int(0.25*fs); tw=int(0.12*fs)
        if ti+tw<n: ecg[ti:ti+tw] += 0.3*np.hanning(tw)
    if cfg['afib']: ecg += 0.05*np.sin(2*np.pi*0.3*np.linspace(0,dur,n))
    ecg += np.random.normal(0,cfg['noise'],n)
    return ecg, fs

model, model_loaded = load_model(str(CHECKPOINT))

# ── Apply theme via JS ─────────────────────────────────────────────────────────
st.markdown(f'''<script>
(function(){{
    var root = window.parent.document.documentElement;
    root.setAttribute("data-theme", "{st.session_state.theme}");
    // Also patch the stApp background directly for Streamlit
    var app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
    if (app) {{
        if ("{st.session_state.theme}" === "light") {{
            app.style.background = "#f8f8fc";
            app.style.color = "#1a1a2e";
        }} else {{
            app.style.background = "#070711";
            app.style.color = "#e2e2f0";
        }}
    }}
}})();
</script>''', unsafe_allow_html=True)


# ── Inject dynamic theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Settings tabs & compact cards */
.stTabs [data-baseweb="tab-panel"] { padding: 1rem 0 !important; }
.settings-metric { font-size: 1.4rem !important; font-weight: 800 !important; }
.settings-card { background: var(--surf) !important; border: 1px solid var(--bdr) !important; border-radius: 10px !important; padding: 1.2rem !important; margin-bottom: 1rem !important; }
.file-dot { font-size: 0.9rem !important; font-weight: bold !important; }
@media (max-width: 768px) {
  .settings-card { padding: 1rem !important; }
  div[data-testid="stMetric"] { text-align: center !important; }
}
</style>""", unsafe_allow_html=True)

if st.session_state.theme == 'light':
    st.markdown("""
    <style>
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stAppViewBlockContainer"],
    section[data-testid="stMain"] {
        background: #f4f4fb !important;
        color: #1a1a2e !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type {
        background: rgba(244,244,251,0.97) !important;
        border-bottom: 1px solid #d8d8ee !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type .stButton>button {
        color: #5a5a8a !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type .stButton>button:hover {
        background: #ebebf8 !important;
        color: #1a1a2e !important;
    }
    .mcard { background: #ffffff !important; border-color: #e0e0f0 !important; }
    .rcard { background: #ffffff !important; border-color: #e0e0f0 !important; }
    .slbl  { color: #9090b0 !important; border-bottom-color: #e0e0f0 !important; }
    .alert.awarn { background: rgba(251,146,60,.06) !important; }
    .alert.aerr  { background: rgba(248,113,113,.06) !important; }
    .alert.aok   { background: rgba(74,222,128,.06) !important; }
    .cbar-track  { background: #e8e8f4 !important; }
    .page-wrap   { background: transparent !important; }
    .hero-title  { color: #1a1a2e !important; }
    .hero-sub    { color: #6a6a9a !important; }
    div[data-testid="stMarkdownContainer"] p { color: #5a5a7a !important; }
    div[data-testid="stExpander"] { background: #ffffff !important; border-color: #e0e0f0 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #ffffff !important; border-color: #e0e0f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #8888aa !important; }
    [data-testid="stFileUploader"] { background: #ffffff !important; border-color: #d0d0e8 !important; }
    code, pre { background: #ebebf8 !important; color: #3a3a6a !important; border-color: #d8d8ee !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* Classify page specific styles */
.page-wrap.classify {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.page-wrap.classify .stImage > img {
    max-height: 500px;
    object-fit: contain;
}
.page-wrap.classify .rcard {
    min-height: 360px;
    border-radius: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.14);
}
.page-wrap.classify .chip {
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
}
.page-wrap.classify .cbar {
    margin-bottom: 1rem;
}
/* Responsive adjustments */
@media (max-width: 768px) {
    .page-wrap.classify {
        padding: 1rem;
    }
    .page-wrap.classify .stImage > img {
        max-height: 300px;
    }
    .page-wrap.classify .rcard {
        /* Stack chips vertically on small screens */
    }
    .page-wrap.classify .chip {
        margin-right: 0;
        margin-bottom: 0.5rem;
        display: block;
    }
}

.page-wrap.simulator {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.page-wrap.simulator .slbl {
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
}
.page-wrap.simulator .stButton > button {
    min-height: 44px;
    font-size: 0.92rem;
}
.page-wrap.simulator .rcard {
    min-height: 315px;
    border-radius: 14px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
}
.page-wrap.simulator .stColumn {
    padding: 0.5rem;
}

@media (max-width: 1024px) {
    .page-wrap.simulator .stImage > img {
        max-height: 340px;
    }
}

@media (max-width: 768px) {
    .page-wrap.simulator .stColumns > div {
        flex: 1 1 100% !important;
    }
    .page-wrap.simulator .slbl {
        margin-top: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR — real Streamlit buttons, styled via CSS ───────────────────────────
cur  = st.session_state.page
cols = st.columns(len(PAGES))
for col, (pid, icon, label) in zip(cols, PAGES):
    with col:
        btn_type = "primary" if cur == pid else "secondary"
        if st.button(f"{icon} {label}", key="nav_"+pid,
                     use_container_width=True, type=btn_type):
            st.session_state.page = pid
            st.rerun()

st.markdown(f'<div class="page-wrap {cur}">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════════
if cur == 'classify':
    st.markdown("""
    <h1 class="hero-title">ECG <span class="hl">Classification</span></h1>
    <p class="hero-sub">Upload a spectrogram image — get a cardiac condition prediction in seconds.</p>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.markdown('<div class="alert aerr">⚠ No checkpoint found. Run the training notebook first.</div>', unsafe_allow_html=True)
    st.markdown('<div class="alert awarn">⚕ RESEARCH TOOL ONLY — Not for clinical use. Consult a qualified cardiologist.</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.4, 1], gap="large")

    with c_left:
        st.markdown('<div class="slbl">Upload Spectrogram</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--dim);font-size:0.85rem;margin-bottom:0.8rem;">Select a spectrogram image and run classification. The model returns top class + confidence breakdown instantly.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("PNG / JPG spectrogram", type=['png','jpg','jpeg'],
                                    label_visibility="collapsed", key="up_classify")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption=uploaded.name)
            if st.button("🔍  Run Classification", use_container_width=True):
                with st.spinner("Analysing..."):
                    pred, probs = classify_pil(img, model)
                    log("Upload", pred, probs, probs.max())
                    st.session_state.last_pred  = pred
                    st.session_state.last_probs = probs.tolist()

    with c_right:
        st.markdown('<div class="slbl">Prediction Result</div>', unsafe_allow_html=True)
        lp = st.session_state.last_pred
        lr = st.session_state.last_probs
        if lp and lr:
            info  = CLASS_INFO[lp]
            probs = np.array(lr)
            conf  = probs.max()*100
            # Build entire result block as one HTML string — no nested st calls
            bhtml = bars_html(probs, lp)
            st.markdown(f"""
            <div class="rcard">
              <div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;
                          color:var(--dimmer);letter-spacing:0.2em;text-transform:uppercase;
                          margin-bottom:0.7rem;">Predicted Condition</div>
              <div class="rclass" style="color:{info['color']};">{info['icon']} {lp}</div>
              <div class="rdesc">{info['desc']}</div>
              <div style="margin-bottom:1rem;">
                <span class="chip" style="background:{info['color']}22;color:{info['color']};border:1px solid {info['color']}44;">
                  ⚡ Risk: {info['risk']}</span>
                <span class="chip" style="background:#7c6af722;color:#a59cfa;border:1px solid #7c6af744;">
                  📊 {conf:.1f}% confident</span>
              </div>
              <div class="slbl" style="margin-top:0;">All Class Scores</div>
              {bhtml}
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            qa1, qa2 = st.columns(2)
            with qa1:
                if st.button("📋  History", use_container_width=True):
                    st.session_state.page='history'; st.rerun()
            with qa2:
                if st.button("📊  Analytics", use_container_width=True):
                    st.session_state.page='analytics'; st.rerun()
        else:
            st.markdown("""
            <div style="border:1.5px dashed #1f1f35;border-radius:10px;
                        padding:3rem 2rem;text-align:center;background:#0f0f1a;">
              <div style="font-size:2rem;margin-bottom:0.7rem;">🫀</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:0.73rem;color:#404060;">
                Upload a spectrogram and click<br>Run Classification
              </div>
            </div>""", unsafe_allow_html=True)

    # ── About Section ──────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("ℹ️  About CardioScan AI", expanded=False):
        a1,a2=st.columns([1.3,1],gap="large")
        with a1:
            st.markdown('<div class="slbl">Project Overview</div>', unsafe_allow_html=True)
            st.markdown('<p>Cardiovascular diseases are a leading cause of death globally. Manual ECG interpretation requires significant clinical expertise. This project reframes ECG analysis as a <b style="color:#e2e2f0;">computer vision problem</b> — ECG beats are converted to 2D spectrograms via STFT and classified by EfficientNetB0.</p>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="slbl">Pipeline</div>', unsafe_allow_html=True)
            for num,title,desc in [
                ("01","Signal Acquisition","MIT-BIH Arrhythmia Database — 48 recordings at 360 Hz"),
                ("02","Beat Segmentation","360-sample windows centered on annotated R-peaks"),
                ("03","STFT Conversion","1D beat → 2D spectrogram (nperseg=64, noverlap=32)"),
                ("04","Transfer Learning","EfficientNetB0 fine-tuned in two phases"),
                ("05","Results","Macro F1=0.9548 · Test Accuracy=97.47%"),
            ]:
                st.markdown(f'<div style="display:flex;gap:1rem;margin-bottom:0.75rem;"><div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#a59cfa;min-width:1.5rem;padding-top:0.05rem;">{num}</div><div><div style="font-size:0.87rem;font-weight:700;color:#e2e2f0;">{title}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.66rem;color:#404060;margin-top:0.1rem;">{desc}</div></div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="slbl">Limitations</div>', unsafe_allow_html=True)
            for lim in ["MIT-BIH has no explicit MI labels — bundle branch blocks used as proxy",
                        "Normal class capped at 20,000 for feasible CPU training",
                        "Single-lead analysis (Lead I only)",
                        "Research tool only — not validated for clinical use"]:
                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.69rem;color:#404060;margin-bottom:0.4rem;padding-left:0.8rem;border-left:2px solid #1f1f35;">· {lim}</div>', unsafe_allow_html=True)

        with a2:
            st.markdown('<div class="slbl">Cardiac Conditions</div>', unsafe_allow_html=True)
            for cls,info in CLASS_INFO.items():
                st.markdown(f'<div style="background:var(--surf);border:1px solid {info["color"]}33;border-left:3px solid {info["color"]};border-radius:10px;padding:1rem 1.1rem;margin-bottom:0.75rem;"><div style="font-size:0.93rem;font-weight:700;color:{info["color"]};margin-bottom:0.3rem;">{info["icon"]} {cls}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#7070a0;line-height:1.7;">{info["desc"]}</div><div style="margin-top:0.5rem;"><span class="chip" style="background:{info["color"]}22;color:{info["color"]};border:1px solid {info["color"]}44;">Risk: {info["risk"]}</span></div></div>', unsafe_allow_html=True)
            st.markdown('<div class="slbl" style="margin-top:0.5rem;">Model Performance</div>', unsafe_allow_html=True)
            for cls,f1 in [('Normal','0.9777'),('MI','0.9893'),('Arrhythmia','0.9716'),('AFib','0.8805')]:
                ci=CLASS_INFO[cls]; pct=float(f1)*100
                st.markdown(f'<div style="margin-bottom:0.65rem;"><div style="display:flex;justify-content:space-between;margin-bottom:0.22rem;"><span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{ci["color"]};">{cls}</span><span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{ci["color"]};font-weight:700;">F1 {f1}</span></div><div style="background:#1f1f35;border-radius:3px;height:4px;"><div style="width:{pct}%;height:4px;border-radius:3px;background:{ci["color"]};"></div></div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif cur == 'simulator':
    st.markdown("""
    <h1 class="hero-title">Signal <span class="hl">Simulator</span></h1>
    <p class="hero-sub">Generate a synthetic ECG, convert to spectrogram, and classify — full pipeline end-to-end.</p>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1.1, 1.4], gap="large")
    with sc1:
        st.markdown('<div class="slbl">Parameters</div>', unsafe_allow_html=True)
        tmpl = st.selectbox("Condition template", list(SYNTH.keys()))
        hr   = st.slider("Heart rate (BPM)", 50, 150, SYNTH[tmpl]['hr'])
        nl   = st.slider("Noise level", 0.0, 0.15, SYNTH[tmpl]['noise'], step=0.01)
        dur  = st.slider("Duration (s)", 1, 5, 2)
        seed = st.number_input("Seed", value=42, step=1)
        go   = st.button("⚡  Generate & Classify", use_container_width=True)
        st.markdown('<div class="alert aok" style="margin-top:0.8rem;">💡 Demonstrates the full ECG → spectrogram → CNN pipeline.</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="slbl">Output</div>', unsafe_allow_html=True)
        if go:
            np.random.seed(int(seed))
            ecg, fs = gen_ecg(tmpl, dur=dur)
            fig,ax  = dkfig(8,2.5)
            ax.plot(np.linspace(0,dur,len(ecg)), ecg, lw=0.9, color='#7c6af7')
            ax.set_title(f'Synthetic ECG — {tmpl}', color='#e2e2f0', fontsize=9)
            ax.set_xlabel('Time (s)', color='#7070a0', fontsize=8)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

            c  = len(ecg)//2; beat = ecg[max(0,c-180):c+180]
            sp = sig2spec(beat, fs=fs)
            fig2,ax2 = dkfig(6,3)
            ax2.imshow(sp, aspect='auto', origin='lower', cmap='plasma')
            ax2.set_title('STFT Spectrogram', color='#e2e2f0', fontsize=9)
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

            pred, probs = classify_pil(spec2pil(sp), model)
            log(f"Simulator ({tmpl})", pred, probs, probs.max())
            info = CLASS_INFO[pred]
            st.markdown(f"""
            <div class="rcard">
              <div class="rclass" style="color:{info['color']};">{info['icon']} {pred}</div>
              <div class="rdesc">Confidence: {probs.max()*100:.1f}%</div>
              {bars_html(probs, pred)}
            </div>""", unsafe_allow_html=True)



elif cur == 'analytics':
    st.markdown("""
    <h1 class="hero-title">Analytics <span class="hl">Dashboard</span></h1>
    <p class="hero-sub">Exploratory data analysis, training diagnostics, and model evaluation results.</p>
    """, unsafe_allow_html=True)

    for col,(lbl,val) in zip(st.columns(5),[
        ('Architecture','EfficientNetB0'),('Parameters','4.01M'),
        ('Dataset','MIT-BIH'),('Classes','4'),('Test Accuracy','97.47%')]):
        col.markdown(f'<div class="mcard"><div class="lbl">{lbl}</div><div class="val" style="font-size:1.05rem;">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["📦 CLASS DIST","📈 LEARNING CURVES","🎯 CONFUSION MATRIX","🔮 CONFIDENCE","🧠 GRAD-CAM","🔬 BEAT ANALYSIS"])
    pairs = [
        ('class_distribution.png','Class distribution','Normal capped at 20,000. Class-weighted CrossEntropyLoss applied.'),
        ('learning_curves.png','Learning curves','Phase 1: head only lr=1e-3. Phase 2: full network lr=1e-4. Best val_loss: 0.0933.'),
        ('confusion_matrix.png','Confusion matrix','Rows = true. Columns = predicted. Diagonal = correct.'),
        ('confidence_distribution.png','Confidence','Skewed toward 1.0 = high confidence on correct predictions.'),
        ('gradcam.png','Grad-CAM','Activations from model.features[-1]. Brighter = higher attention.'),
    ]
    for tab,(fname,cap,note) in zip(tabs,pairs):
        p = OUTPUTS_DIR/fname
        if p.exists():
            tab.image(str(p),caption=cap,use_container_width=True)
        else:
            tab.markdown(f'<div class="alert aerr">⚠ {fname} not found in ecg_project/outputs/</div>', unsafe_allow_html=True)
        tab.markdown(f'<p>{note}</p>', unsafe_allow_html=True)

    # Beat Analysis tab
    with tabs[5]:
        st.markdown('<div class="slbl">Beat-by-Beat Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p>Upload a raw ECG CSV to classify every beat and view an annotated rhythm timeline.</p>', unsafe_allow_html=True)
        st.markdown('<div class="alert awarn">⚕ For demonstration purposes only. Not for clinical use.</div>', unsafe_allow_html=True)

        b1, b2 = st.columns([1,1.6], gap="large")
        with b1:
            st.markdown('<div class="slbl">Input Signal</div>', unsafe_allow_html=True)
            csv_f    = st.file_uploader("ECG CSV (one column, 360 Hz)", type=['csv','txt'], label_visibility="collapsed")
            use_demo = st.button("📥  Use Demo Signal", use_container_width=True)
            fs_in    = st.number_input("Sampling rate (Hz)", value=360, step=1)
            analyse  = st.button("🔬  Analyse Beats", use_container_width=True)
            st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#404060;margin-top:0.8rem;line-height:1.8;"><b style="color:#7070a0;">CSV format:</b><br>One value per row · 360 Hz · No header</div>', unsafe_allow_html=True)

        with b2:
            st.markdown('<div class="slbl">Beat Timeline</div>', unsafe_allow_html=True)
            raw = None
            if csv_f:
                try:
                    vals = [float(x.strip().split(',')[0]) for x in csv_f.read().decode().strip().split('\n') if x.strip()]
                    raw  = np.array(vals)
                    st.markdown(f'<div class="alert aok">✅ Loaded {len(raw):,} samples ({len(raw)/int(fs_in):.1f}s)</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="alert aerr">❌ {e}</div>', unsafe_allow_html=True)
            elif use_demo:
                np.random.seed(7)
                s1,_ = gen_ecg('Normal',     duration_sec=3)
                s2,_ = gen_ecg('Arrhythmia', duration_sec=2)
                s3,_ = gen_ecg('Normal',     duration_sec=2)
                raw  = np.concatenate([s1,s2,s3])
                st.markdown(f'<div class="alert aok">✅ Demo: Normal 3s → Arrhythmia 2s → Normal 2s ({len(raw):,} samples)</div>', unsafe_allow_html=True)

            if raw is not None and analyse:
                fs  = int(fs_in); hb = int(0.25*fs); rp = []; i = hb
                while i < len(raw)-hb:
                    w = raw[i-hb:i+hb]
                    if raw[i]==w.max() and raw[i]>0.3*w.max(): rp.append(i); i+=hb
                    else: i+=1
                if len(rp)<2:
                    st.markdown('<div class="alert aerr">⚠ Not enough beats detected.</div>', unsafe_allow_html=True)
                else:
                    with st.spinner(f"Classifying {len(rp)} beats..."):
                        bpreds,bconfs,bprobs=[],[],[]
                        for r in rp:
                            s,e=r-180,r+180
                            if s<0 or e>len(raw):
                                bpreds.append('Normal');bconfs.append(0.5);bprobs.append([0.25]*4);continue
                            p,pr=classify_pil(spec2pil(sig2spec(raw[s:e],fs=fs)),model)
                            bpreds.append(p);bconfs.append(pr.max());bprobs.append(pr.tolist())
                    counts=Counter(bpreds); total=len(bpreds)
                    mc=st.columns(4)
                    for m,cls in zip(mc,CLASS_NAMES):
                        ci=CLASS_INFO[cls]
                        m.markdown(f'<div class="mcard" style="border-left:3px solid {ci["color"]};"><div class="lbl">{cls}</div><div class="val" style="color:{ci["color"]};font-size:1.3rem;">{counts.get(cls,0)}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#404060;">{counts.get(cls,0)/total*100:.0f}% of beats</div></div>', unsafe_allow_html=True)
                    cmap={k:v['color'] for k,v in CLASS_INFO.items()}
                    fig,axes=dkfigs(2,1,w=10,h=5,gridspec_kw={'height_ratios':[2,1]})
                    axes[0].plot(raw,lw=0.6,color='#2a2a45',alpha=0.9)
                    for r,p in zip(rp,bpreds): axes[0].axvline(x=r,color=cmap[p],lw=1.2,alpha=0.7)
                    axes[0].set_title('ECG Signal with Beat Classifications',color='#e2e2f0',fontsize=9)
                    axes[1].plot(range(len(bconfs)),bconfs,'o-',color='#7c6af7',ms=3,lw=0.9)
                    axes[1].axhline(y=0.5,color='#2a2a45',ls='--',lw=0.8)
                    axes[1].set_title('Model Confidence per Beat',color='#e2e2f0',fontsize=9)
                    axes[1].set_ylim(0,1)
                    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
                    log("Beat Analysis",f"{counts.most_common(1)[0][0]} (dominant)",
                        np.array(bprobs).mean(axis=0),np.array(bconfs).mean())

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="slbl">Classification Report</div>', unsafe_allow_html=True)
    rp = OUTPUTS_DIR/'classification_report.txt'
    if rp.exists(): st.code(open(rp).read(), language=None)
    else: st.markdown('<div class="alert aerr">⚠ Run Cell 19 in the notebook to generate this.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif cur == 'explorer':
    st.markdown("""
    <h1 class="hero-title">Class <span class="hl">Explorer</span></h1>
    <p class="hero-sub">Browse real MIT-BIH spectrogram images by class.</p>
    """, unsafe_allow_html=True)

    e1,e2 = st.columns([1,2.5],gap="large")
    with e1:
        st.markdown('<div class="slbl">Filter</div>', unsafe_allow_html=True)
        sel  = st.selectbox("Condition", CLASS_NAMES)
        nsmp = st.slider("Images to show", 4, 24, 12, step=4)
        ci   = CLASS_INFO[sel]
        st.markdown(f'<div class="mcard" style="border-left:3px solid {ci["color"]};margin-top:0.8rem;"><div class="lbl">Selected</div><div class="val" style="color:{ci["color"]};font-size:1.2rem;">{ci["icon"]} {sel}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#7070a0;margin-top:0.4rem;line-height:1.7;">{ci["desc"]}</div></div>', unsafe_allow_html=True)
    with e2:
        cd = SPEC_DIR/sel
        if cd.exists():
            af = list(cd.glob('*.png'))
            if af:
                random.seed(42); samples = random.sample(af, min(nsmp,len(af)))
                st.markdown(f'<div class="slbl">{sel} — {len(af):,} total · showing {len(samples)}</div>', unsafe_allow_html=True)
                gc = st.columns(4)
                for i,fp in enumerate(samples):
                    with gc[i%4]:
                        img_e = Image.open(fp)
                        st.image(img_e,caption=fp.stem,use_container_width=True)
                        if st.button("Classify",key=f"ex{i}"):
                            pe,pre=classify_pil(img_e,model)
                            log(f"Explorer ({sel})",pe,pre,pre.max())
                            rc=CLASS_INFO[pe]['color']
                            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.63rem;color:{rc};">→ {pe} ({pre.max()*100:.0f}%)</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="alert aerr">⚠ No images found. Run Cell 8 in the notebook.</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="alert aerr">⚠ ecg_project/ecg_spectrograms/ not found.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif cur == 'compare':
    st.markdown("""
    <h1 class="hero-title">Model <span class="hl">Comparison</span></h1>
    <p class="hero-sub">Phase 1 (head only) vs Phase 2 (fully fine-tuned) on the same image.</p>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_p1():
        m=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        m.classifier=nn.Sequential(nn.Dropout(0.4,inplace=True),nn.Linear(m.classifier[1].in_features,4))
        m.eval(); return m

    cm1,cm2,cm3 = st.columns(3,gap="large")
    with cm1:
        st.markdown('<div class="slbl">Upload Image</div>', unsafe_allow_html=True)
        cup = st.file_uploader("Spectrogram",type=['png','jpg','jpeg'],label_visibility="collapsed")
        if cup: cimg=Image.open(cup); st.image(cimg,use_container_width=True)
        cbtn = st.button("⚖️  Compare",use_container_width=True) if cup else False

    with cm2:
        st.markdown('<div class="slbl">Phase 1 — Head Only</div>', unsafe_allow_html=True)
        st.markdown('<div class="mcard"><div class="lbl">Config</div><div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#7070a0;line-height:1.8;">Backbone: frozen (ImageNet)<br>Trained: Linear(1280→4)<br>Epochs: 5 · lr=1e-3</div></div>', unsafe_allow_html=True)
        if cup and cbtn:
            p1p,p1pr=classify_pil(cimg,load_p1())
            i1=CLASS_INFO[p1p]
            st.markdown(f'<div class="rcard"><div class="rclass" style="color:{i1["color"]};">{i1["icon"]} {p1p}</div><div class="rdesc">Confidence: {p1pr.max()*100:.1f}%</div>{bars_html(p1pr,p1p)}</div>', unsafe_allow_html=True)

    with cm3:
        st.markdown('<div class="slbl">Phase 2 — Full Fine-Tune</div>', unsafe_allow_html=True)
        st.markdown('<div class="mcard"><div class="lbl">Config</div><div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#7070a0;line-height:1.8;">Backbone: unfrozen (ECG-adapted)<br>Trained: all 4M params<br>Epochs: 11 · Best loss: 0.0933</div></div>', unsafe_allow_html=True)
        if cup and cbtn:
            if model_loaded:
                p2p,p2pr=classify_pil(cimg,model)
                i2=CLASS_INFO[p2p]; delta=p2pr.max()-p1pr.max(); dc='#4ade80' if delta>=0 else '#f87171'
                st.markdown(f'<div class="rcard"><div class="rclass" style="color:{i2["color"]};">{i2["icon"]} {p2p}</div><div class="rdesc">Confidence: {p2pr.max()*100:.1f}%<br><span style="color:{dc};">Δ vs Phase 1: {delta*100:+.1f}%</span></div>{bars_html(p2pr,p2p)}</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="alert aerr">⚠ Phase 2 checkpoint not found.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif cur == 'history':
    st.markdown("""
    <h1 class="hero-title">Prediction <span class="hl">History</span></h1>
    <p class="hero-sub">Session log of all classifications across every page.</p>
    """, unsafe_allow_html=True)

    hist = st.session_state.pred_history
    if not hist:
        st.markdown('<div style="border:1.5px dashed #1f1f35;border-radius:10px;padding:3rem 2rem;text-align:center;background:#0f0f1a;"><div style="font-size:2rem;margin-bottom:0.7rem;">📋</div><div style="font-family:JetBrains Mono,monospace;font-size:0.73rem;color:#404060;">No predictions yet. Use any page to get started.</div></div>', unsafe_allow_html=True)
    else:
        hc,bc = st.columns([3,1])
        hc.markdown(f'<p>{len(hist)} predictions this session</p>', unsafe_allow_html=True)
        with bc:
            if st.button("🗑  Clear"): st.session_state.pred_history=[]; st.rerun()

        preds=Counter([h['prediction'] for h in hist])
        for col,cls in zip(st.columns(4),CLASS_NAMES):
            ci=CLASS_INFO[cls]
            col.markdown(f'<div class="mcard" style="border-left:3px solid {ci["color"]};"><div class="lbl">{cls}</div><div class="val" style="color:{ci["color"]};font-size:1.3rem;">{preds.get(cls,0)}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slbl">Session Log</div>', unsafe_allow_html=True)
        hdrs=['Time','Source','Prediction','Confidence','AFib','Arrhy.','MI','Normal']
        hcols=st.columns([0.7,1.6,1.1,0.9,0.7,0.7,0.7,0.7])
        for c,h in zip(hcols,hdrs):
            c.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:#404060;letter-spacing:0.1em;padding-bottom:0.4rem;border-bottom:1px solid #1f1f35;">{h}</div>', unsafe_allow_html=True)
        for entry in reversed(hist):
            rc=st.columns([0.7,1.6,1.1,0.9,0.7,0.7,0.7,0.7])
            pc=CLASS_INFO.get(entry['prediction'].split(' ')[0],{}).get('color','#e2e2f0')
            for c,v,cl in zip(rc,
                [entry['time'],entry['source'],entry['prediction'],entry['confidence'],
                 entry['AFib'],entry['Arrhythmia'],entry['MI'],entry['Normal']],
                ['#7070a0','#9090b0',pc,'#e2e2f0','#fb923c','#f87171','#f472b6','#4ade80']):
                c.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.64rem;color:{cl};padding:0.28rem 0;border-bottom:1px solid #0f0f1a;">{v}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥  Export CSV"):
            buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=hist[0].keys())
            w.writeheader(); w.writerows(hist)
            st.download_button("Download",data=buf.getvalue(),
                file_name=f"cardioscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
elif cur == 'howitworks':
    st.markdown("""
    <h1 class="hero-title">How It <span class="hl">Works</span></h1>
    <p class="hero-sub">Interactive walkthrough of the full ECG → prediction pipeline.</p>
    """, unsafe_allow_html=True)

    steps=[
        ("01","Raw ECG Signal","An Electrocardiogram records heart electrical activity as a 1D waveform at 360 Hz. Each cardiac cycle shows P-QRS-T morphology.","record = wfdb.rdrecord('mitdb/100')\nsignal = record.p_signal[:, 0]  # Lead I\n# Shape: (650,000,) — 30 min at 360 Hz"),
        ("02","Beat Segmentation","360-sample windows (1 second) centered on annotated R-peaks capture each complete cardiac cycle.","beat = signal[r_peak - 180 : r_peak + 180]\n# Shape: (360,) — 1 second at 360 Hz"),
        ("03","STFT Spectrogram","Short-Time Fourier Transform converts the 1D beat to a 2D time-frequency image. Different conditions produce distinct spectral signatures.","_, _, Zxx = scipy.signal.stft(beat, fs=360, nperseg=64)\nspec = 20 * log10(|Zxx| + eps)   # dB\nspec = (spec - min) / range      # [0,1]"),
        ("04","Transfer Learning","EfficientNetB0 pre-trained on ImageNet already detects edges and textures. We adapt these features to ECG spectrogram patterns.","model = efficientnet_b0(weights='DEFAULT')\nmodel.classifier = nn.Sequential(\n    nn.Dropout(0.4),\n    nn.Linear(1280, 4))"),
        ("05","Two-Phase Training","Phase 1: head only (5 epochs, lr=1e-3). Phase 2: all 4M params (11 epochs, lr=1e-4). Best val_loss: 0.0933, val_acc: 97.75%.","# Phase 1: 5,124 trainable params\n# Phase 2: 4,012,672 trainable params\n# Early stopping at epoch 14"),
        ("06","Prediction","Softmax converts 4 logits to probabilities. Grad-CAM shows which spectrogram regions drove the decision.","probs = F.softmax(model(tensor), dim=1)\npred  = CLASS_NAMES[probs.argmax()]\n# e.g. MI with 91.6% confidence"),
    ]
    for i,(num,title,desc,code) in enumerate(steps):
        with st.expander(f"Step {num} — {title}", expanded=(i==0)):
            d,c=st.columns([1.3,1],gap="large")
            d.markdown(f'<p>{desc}</p>', unsafe_allow_html=True)
            c.code(code,language='python')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="slbl">Live Pipeline Demo</div>', unsafe_allow_html=True)
    dc1,dc2=st.columns([1,3],gap="large")
    with dc1:
        dcls=st.selectbox("Condition",CLASS_NAMES,key="hwdemo")
        dbtn=st.button("▶  Run Demo",use_container_width=True)
    with dc2:
        if dbtn:
            np.random.seed(42); ecg,fs=gen_ecg(dcls)
            fig,ax=dkfig(9,2.5)
            ax.plot(np.linspace(0,2,len(ecg)),ecg,lw=0.9,color='#7c6af7')
            ax.set_title(f'Step 1 — Raw ECG ({dcls})',color='#e2e2f0',fontsize=9)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
            beat=ecg[len(ecg)//2-180:len(ecg)//2+180]; spec=sig2spec(beat)
            fig2,ax2=dkfig(6,3)
            ax2.imshow(spec,aspect='auto',origin='lower',cmap='plasma')
            ax2.set_title('Step 2+3 — Beat → Spectrogram',color='#e2e2f0',fontsize=9)
            plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close(fig2)
            pred,probs=classify_pil(spec2pil(spec),model)
            log(f"How It Works ({dcls})",pred,probs,probs.max())
            info=CLASS_INFO[pred]
            st.markdown(f'<div class="rcard"><div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:var(--dimmer);letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.5rem;">Step 4 — Prediction</div><div class="rclass" style="color:{info["color"]};">{info["icon"]} {pred}</div><div class="rdesc">Confidence: {probs.max()*100:.1f}%</div>{bars_html(probs,pred)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

elif cur == 'settings':
    st.markdown("""
    <h1 class="hero-title">App <span class="hl">Settings</span></h1>
    <p class="hero-sub">Customise appearance, inspect model info, and manage your session.</p>
    """, unsafe_allow_html=True)

    tab_app, = st.tabs([
        "🎨  Appearance"
    ])

    # ── Appearance ─────────────────────────────────────────────────────────────
    with tab_app:
        st.markdown('<div class="slbl">Theme</div>', unsafe_allow_html=True)
        cur_theme = st.session_state.theme
        tc1, tc2 = st.columns(2, gap="medium")
        with tc1:
            is_dark  = cur_theme == 'dark'
            bd       = '#7c6af7' if is_dark else '#1f1f35'
            ck       = '✓ ' if is_dark else ''
            st.markdown(f"""
            <div style="background:#0f0f1a;border:2px solid {bd};border-radius:12px;
                        padding:1.4rem;text-align:center;margin-bottom:0.6rem;">
              <div style="font-size:1.8rem;margin-bottom:0.4rem;">🌙</div>
              <div style="font-size:0.88rem;font-weight:700;color:#e2e2f0;">{ck}Dark Mode</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:0.63rem;color:#7070a0;margin-top:0.3rem;">
                Deep navy · Purple accents</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Select Dark", use_container_width=True, key="set_dark",
                         type="primary" if is_dark else "secondary"):
                st.session_state.theme = 'dark'
                st.rerun()
        with tc2:
            is_light = cur_theme == 'light'
            bl       = '#7c6af7' if is_light else '#e0e0f0'
            cl       = '✓ ' if is_light else ''
            st.markdown(f"""
            <div style="background:#ffffff;border:2px solid {bl};border-radius:12px;
                        padding:1.4rem;text-align:center;margin-bottom:0.6rem;">
              <div style="font-size:1.8rem;margin-bottom:0.4rem;">☀️</div>
              <div style="font-size:0.88rem;font-weight:700;color:#1a1a2e;">{cl}Light Mode</div>
              <div style="font-family:JetBrains Mono,monospace;font-size:0.63rem;color:#7070a0;margin-top:0.3rem;">
                Clean white · Purple accents</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Select Light", use_container_width=True, key="set_light",
                         type="primary" if is_light else "secondary"):
                st.session_state.theme = 'light'
                st.rerun()
        active_label = "Dark" if cur_theme == "dark" else "Light"
        st.markdown(f'<div class="alert aok" style="margin-top:0.5rem;">✓ Active: <b>{active_label} Mode</b></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
