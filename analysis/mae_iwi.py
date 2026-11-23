# Usage: python mae_iwi.py <IWI eval_test_predictions.csv.gz> <out.csv>
# IWI: nine-model mean change in MAE and RMSE per encoding (all, rural, urban, DMSP-era, VIIRS-era), paired Poisson bootstrap
import sys, numpy as np, pandas as pd
P = pd.read_csv(sys.argv[1])
TGT = 'iwi'; BASE = 'L+NL'
NINE = ['resnet18','resnet34','resnet50','swin','unet','vit','satclip','mosaiks','olmoearth']
ENC = ['L+NL+wavelet','L+NL+DFT','L+NL+peak:pixel11/wavelet6','L+NL+TT:cyclic']
y = P[TGT].to_numpy(float)
g = pd.factorize(P['group'])[0]
W = np.random.default_rng(20261117).poisson(1.0, size=(1000, g.max() + 1)).astype(float)[:, g]
rural = P['rural'].astype(str).str.lower().eq('true').to_numpy()
subs = {'all': np.ones(len(P), bool), 'rural': rural, 'urban': ~rural,
        'DMSP': (P['nl_sensor'] == 'DMSP').to_numpy(), 'VIIRS': (P['nl_sensor'] == 'VIIRS').to_numpy()}
rows = []
for start in ['scratch', 'pretrained']:
    for enc in ENC:
        for sn, sub in subs.items():
            dmae, rel, bm, dr2, br2 = [], [], [], [], []
            for m in NINE:
                a = P[f'ten_models|{m}|{enc}|{start}'].to_numpy(float); b = P[f'ten_models|{m}|{BASE}|{start}'].to_numpy(float)
                k = sub & np.isfinite(a) & np.isfinite(b)
                ea, eb = np.abs(y[k]-a[k]), np.abs(y[k]-b[k])
                dmae.append(ea.mean()-eb.mean()); rel.append(dmae[-1]/eb.mean())
                w = W[:, k]; bm.append((w@ea - w@eb)/w.sum(1))
                sst = ((y[k]-y[k].mean())**2).sum()
                dr2.append((((y[k]-b[k])**2).sum() - ((y[k]-a[k])**2).sum())/sst)
            bm = np.mean(bm, 0)
            rows.append(dict(start=start, enc=enc.replace('L+NL+',''), subset=sn, dMAE=np.mean(dmae), rel=np.mean(rel),
                             lo=np.percentile(bm,2.5), hi=np.percentile(bm,97.5), dR2=np.mean(dr2)))
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], index=False)
pd.set_option('display.width', 200)
print(out.to_string(index=False, float_format=lambda v: '%.4f' % v))
