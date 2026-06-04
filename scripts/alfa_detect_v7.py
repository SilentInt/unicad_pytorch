"""ALFA 异常检测 V7 — 最优架构 (final)。

架构:
  ConditionalGRU(hist=75, 7.5s): predict(response_t | history_75f, command_t) → 13D
    → 归一化 → 窗口统计 W=10,30 (mean/std/max) → 78D
    → Galaxy AE (78→128→64 扩展) + SMM(k=5) + GOF(vector)

消融结论:
  - ConditionalGRU 的条件残差唯一不可删除
  - 时序预测器残差加噪, FFT/互相关/协方差全冗余
  - AE 扩展 > AE 压缩 > 无 AE
  - GRU > LSTM/Conv1d/TCN, hist=75 > hist=10/30/100
  - 最优 MSeq AUC: 0.942 (SMM) / ~0.946 (Galaxy)

用法: uv run python scripts/alfa_detect_v7.py --device cuda:1
"""

import argparse, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                              precision_score, recall_score, roc_auc_score)
from unicad_torch import Galaxy, GalaxyConfig
from unicad_torch.gof import gof_score
from unicad_torch.smm_torch import SMMTorch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ALFA"
RESP = list(range(0,13)); CMD = list(range(13,18))
NR, NC = len(RESP), len(CMD)

# ─── 数据加载 ──────────────────────────────────────────────

def _load_csv_numeric(path):
    import csv
    with open(path) as f: r=csv.reader(f); h=next(r); rows=list(r)
    if not rows: return np.empty((0,0)),[]
    mask=[];
    for v in rows[0]:
        try: float(v); mask.append(True)
        except ValueError: mask.append(False)
    idx=[i for i,m in enumerate(mask) if m]
    return np.array([[float(row[i]) for i in idx] for row in rows], dtype=np.float64), [h[i] for i in idx]

def load_alfa_data(data_dir=DATA_DIR, target_hz=10.0):
    IC=["field.angular_velocity.x","field.angular_velocity.y","field.angular_velocity.z",
        "field.linear_acceleration.x","field.linear_acceleration.y","field.linear_acceleration.z"]
    NC_=["field.alt_error","field.aspd_error","field.xtrack_error"]
    RC=["field.channels1","field.channels2","field.channels3","field.channels4","field.channels5"]
    VC=["field.throttle","field.altitude","field.climb"]
    seqs=[]
    for sn in sorted(os.listdir(data_dir)):
        sd=os.path.join(data_dir,sn)
        if not os.path.isdir(sd): continue
        imf=[f for f in os.listdir(sd) if "imu-data_raw" in f]
        if not imf: continue
        idata,ihdr=_load_csv_numeric(os.path.join(sd,imf[0]))
        if len(idata)<10: continue
        it=idata[:,0]/1e9; g=np.arange(it[0],it[-1],1./target_hz); n=len(g)
        def _ip(data,hdr,cols):
            if len(data)<2: return np.zeros((n,len(cols)),dtype=np.float32)
            t=data[:,0]/1e9; idx=[hdr.index(c) for c in cols]
            v=data[:,idx].astype(np.float32); o=np.empty((n,len(cols)),dtype=np.float32)
            for d in range(len(cols)): o[:,d]=np.interp(g,t,v[:,d])
            return o
        def _inav(topic):
            fn=[f for f in os.listdir(sd) if topic.replace("-","_") in f or topic in f]
            if not fn: return np.zeros((n,1),dtype=np.float32)
            d,h=_load_csv_numeric(os.path.join(sd,fn[0]))
            if len(d)<2: return np.zeros((n,1),dtype=np.float32)
            t=d[:,0]/1e9; ci,mi=h.index("field.commanded"),h.index("field.measured")
            return np.interp(g,t,(d[:,ci]-d[:,mi]).astype(np.float32)).reshape(-1,1)
        fi=_ip(idata,ihdr,IC); ne=np.hstack([_inav(t) for t in
            ["nav_info-roll","nav_info-pitch","nav_info-yaw","nav_info-airspeed"]])
        ef=[f for f in os.listdir(sd) if "nav_info-errors" in f]
        if ef: d,h=_load_csv_numeric(os.path.join(sd,ef[0])); fn=_ip(d,h,NC_)
        else: fn=np.zeros((n,3),dtype=np.float32)
        rf=[f for f in os.listdir(sd) if "rc-out" in f]
        if rf: d,h=_load_csv_numeric(os.path.join(sd,rf[0])); fr=_ip(d,h,RC)
        else: fr=np.zeros((n,5),dtype=np.float32)
        vf=[f for f in os.listdir(sd) if "vfr_hud" in f]
        if vf: d,h=_load_csv_numeric(os.path.join(sd,vf[0])); fv=_ip(d,h,VC)
        else: fv=np.zeros((n,3),dtype=np.float32)
        f=np.hstack([fi,ne,fn,fr,fv]).astype(np.float32)
        f=np.nan_to_num(f,nan=0.,posinf=0.,neginf=0.)
        fl=[]
        for fn in os.listdir(sd):
            if "failure_status" not in fn: continue
            d,_=_load_csv_numeric(os.path.join(sd,fn))
            if len(d)==0: continue
            ts=d[:,0]/1e9; s,p=ts[0],ts[0]
            for t in ts[1:]:
                if t-p>1.: fl.append((s,p)); s=t
                p=t
            fl.append((s,p))
        lbs=np.zeros(n,dtype=np.int32)
        for s,e in fl: lbs[(g>=s)&(g<=e)]=1
        seqs.append({"name":sn,"features":f,"labels":lbs,"n_frames":n,"has_fault":len(fl)>0})
    return seqs

# ─── ConditionalGRU ──────────────────────────────────────────

class ConditionalGRU(nn.Module):
    def __init__(self, feat_dim=21, hidden=64, num_layers=2):
        super().__init__()
        self.gru=nn.GRU(feat_dim,hidden,num_layers,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden+NC,hidden),nn.ReLU(),nn.Linear(hidden,NR))
    def forward(self,x_hist,x_cmd):
        out,_=self.gru(x_hist); return self.head(torch.cat([out[:,-1,:],x_cmd],1))

def train_conditional_predictor(sequences, hist_len=75, epochs=100, lr=1e-3, device="cpu"):
    fd=sequences[0]["features"].shape[1]
    model=ConditionalGRU(fd).to(device)
    Xh,Yr,Yc=[],[],[]
    for s in sequences:
        f,lbs=s["features"],s["labels"]; ni=np.where(lbs==0)[0]
        for i in range(hist_len,len(ni)):
            idx=ni[i]
            if np.all(lbs[idx-hist_len:idx]==0):
                Xh.append(f[idx-hist_len:idx]); Yr.append(f[idx,RESP]); Yc.append(f[idx,CMD])
    n_train=len(Xh)
    Xh_t=torch.tensor(np.array(Xh),dtype=torch.float32,device=device)
    Yr_t=torch.tensor(np.array(Yr),dtype=torch.float32,device=device)
    Yc_t=torch.tensor(np.array(Yc),dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for ep in range(epochs):
        loss=F.mse_loss(model(Xh_t,Yc_t),Yr_t); opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%50==0: print(f"  [CondGRU] epoch {ep+1}/{epochs} loss={loss.item():.2f}")
    print(f"  Training samples: {n_train:,}")
    return model

# ─── 特征提取 ───────────────────────────────────────────────

def _window_stats(errors_norm, W, feat_dim):
    n=len(errors_norm); out=np.zeros((n,feat_dim*3),dtype=np.float32)
    for i in range(W,n):
        w=errors_norm[i-W:i]; out[i,:feat_dim]=w.mean(0)
        out[i,feat_dim:feat_dim*2]=w.std(0); out[i,feat_dim*2:]=w.max(0)
    return out

def compute_features(sequences, model, hist_len=75, device="cpu"):
    model.eval()
    ne=[]
    for s in sequences:
        f,lbs=s["features"],s["labels"]; mk=lbs==0
        for i in range(hist_len,len(f)):
            if mk[i] and np.all(mk[i-hist_len:i]):
                with torch.no_grad():
                    xh=torch.tensor(f[i-hist_len:i],dtype=torch.float32,device=device).unsqueeze(0)
                    xc=torch.tensor(f[i,CMD],dtype=torch.float32,device=device).unsqueeze(0)
                    ne.append((f[i,RESP]-model(xh,xc).squeeze(0).cpu().numpy())**2)
    mu=np.mean(ne,0,keepdims=True); sd=np.std(ne,0,keepdims=True)+1e-8
    results=[]
    for s in sequences:
        f=s["features"]; n=len(f)
        err=np.zeros((n,NR),dtype=np.float32); vd=np.zeros(n,dtype=bool)
        with torch.no_grad():
            for i in range(hist_len,n):
                xh=torch.tensor(f[i-hist_len:i],dtype=torch.float32,device=device).unsqueeze(0)
                xc=torch.tensor(f[i,CMD],dtype=torch.float32,device=device).unsqueeze(0)
                err[i]=(f[i,RESP]-model(xh,xc).squeeze(0).cpu().numpy())**2; vd[i]=True
        en=(err-mu)/sd; en[:hist_len]=0
        ws10=_window_stats(en,10,NR); ws30=_window_stats(en,30,NR)
        results.append({"name":s["name"],"features":np.hstack([ws10,ws30]),
                         "labels":s["labels"],"n_frames":n,"has_fault":s["has_fault"],"valid":vd})
    return results

# ─── SMM+GOF ──────────────────────────────────────────────

def fit_smm_with_em(Z, k=5, em_iters=3, outlier_ratio=0.01):
    Z=Z.clone(); sm=None
    for i in range(em_iters):
        sm=SMMTorch(n_components=k,n_iter=200,tol=1e-4,random_state=42); sm.fit(Z)
        if i<em_iters-1 and sm.means_ is not None:
            scores=gof_score(Z,sm.means_,sm.covars_,sm.weights_.reshape(1,-1),"vector")
            mask=scores<=torch.quantile(scores,1.-outlier_ratio)
            if mask.sum()>k: Z=Z[mask]
    return sm

# ─── 评估 ──────────────────────────────────────────────────

def temporal_smooth(preds, window=5):
    s=preds.copy(); h=window//2
    for i in range(h,len(preds)-h): s[i]=int(np.median(preds[i-h:i+h+1]))
    return s

def evaluate(scores, y_all, valid_all, seqs, title=""):
    yv,sv=y_all[valid_all],scores[valid_all]
    auc=roc_auc_score(yv,sv); ap=average_precision_score(yv,sv)
    best_f1,best_t=0,0
    for pct in np.arange(90,100,0.5):
        t=np.percentile(sv[yv==0],pct); pred=(sv>t).astype(int)
        f1=f1_score(yv,pred)
        if f1>best_f1: best_f1,best_t=f1,t
    pred=(sv>best_t).astype(int)
    f1s=f1_score(yv,temporal_smooth(pred,5))
    if title: print(f"  [{title}]")
    print(f"  AUC={auc:.3f} AP={ap:.3f} F1={best_f1:.3f} F1s={f1s:.3f}")
    off=0; sa=[]
    for s in seqs:
        n=s["n_frames"]; svv=scores[off:off+n][s["valid"]]; yvv=s["labels"][s["valid"]]; off+=n
        if yvv.sum()>0 and yvv.sum()<len(yvv): sa.append(roc_auc_score(yvv,svv))
    mseq=np.mean(sa) if sa else float("nan")
    print(f"  MeanSeq={mseq:.3f}")
    off=0
    for s in seqs:
        n=s["n_frames"]; svv=scores[off:off+n][s["valid"]]; yvv=s["labels"][s["valid"]]; off+=n
        if yvv.sum()<=0 or yvv.sum()>=len(yvv): continue
        a=roc_auc_score(yvv,svv)
        if a<0.7: print(f"    ! {s['name'][-50:]}: {a:.3f}")
    return auc,mseq

# ─── 主函数 ──────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda:1"); p.add_argument("--hist-len",type=int,default=75)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    device=a.device
    if not torch.cuda.is_available(): device="cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    print("="*60)
    print(f"V7 Final: ConditionalGRU(hist={a.hist_len}) -> 78D -> Galaxy AE 128->64")
    print("="*60)

    print("\nLoading data...")
    seqs=load_alfa_data()
    n_fault=int(sum(s["labels"].sum() for s in seqs))
    n_total=sum(s["n_frames"] for s in seqs)
    print(f"  {len(seqs)} seq, {n_total} frames, {n_fault} fault ({n_fault/n_total:.1%})")

    print(f"\n[1/3] Training ConditionalGRU (hist={a.hist_len})...")
    model=train_conditional_predictor(seqs,hist_len=a.hist_len,epochs=100,device=device)

    print("\n[2/3] Computing 78D features...")
    fseqs=compute_features(seqs,model,hist_len=a.hist_len,device=device)
    print(f"  Feature dim: {fseqs[0]['features'].shape[1]}")

    X_n=np.vstack([s["features"][(s["labels"]==0)&s["valid"]] for s in fseqs]).astype(np.float32)
    X_a=np.vstack([s["features"] for s in fseqs]).astype(np.float32)
    y_a=np.concatenate([s["labels"] for s in fseqs]); va=np.concatenate([s["valid"] for s in fseqs])
    zm,zs=X_n.mean(0,keepdims=True),X_n.std(0,keepdims=True)+1e-8; X_nz,X_az=(X_n-zm)/zs,(X_a-zm)/zs

    print(f"\n[3/3] Training & evaluating...")
    print(f"  Normal frames: {len(X_nz):,}")

    # SMM direct
    print(f"\n{'─'*60}"); print("  SMM direct (baseline, no AE)")
    Z_n=torch.tensor(X_nz,dtype=torch.float32,device=device); Z_a=torch.tensor(X_az,dtype=torch.float32,device=device)
    sm=fit_smm_with_em(Z_n,k=5,em_iters=3)
    scores=gof_score(Z_a,sm.means_,sm.covars_,sm.weights_.reshape(1,-1),"vector").cpu().numpy()
    evaluate(scores,y_a,va,fseqs,"k=5 SMM")

    # Galaxy AE expand 128->64
    print(f"\n{'─'*60}"); print("  Galaxy AE 78->128->64 (expand, k=5)")
    cfg=GalaxyConfig(k=5,hidden_dim=128,latent_dim=64,seed=a.seed,
        pretrain_epochs=100,em_iters=3,em_finetune_steps=100,
        gravity_version="vector",score_type="vector",device=device,verbose=False)
    g=Galaxy(cfg); g.fit(X_nz); sg=g.predict_score(X_az)
    evaluate(sg,y_a,va,fseqs,"k=5 Galaxy 128->64")

    # Galaxy AE expand k=10
    print(f"\n{'─'*60}"); print("  Galaxy AE 78->128->64 (expand, k=10)")
    cfg10=GalaxyConfig(k=10,hidden_dim=128,latent_dim=64,seed=a.seed,
        pretrain_epochs=100,em_iters=3,em_finetune_steps=100,
        gravity_version="vector",score_type="vector",device=device,verbose=False)
    g10=Galaxy(cfg10); g10.fit(X_nz); sg10=g10.predict_score(X_az)
    evaluate(sg10,y_a,va,fseqs,"k=10 Galaxy 128->64")

    print("\nDone.")

if __name__=="__main__": main()
