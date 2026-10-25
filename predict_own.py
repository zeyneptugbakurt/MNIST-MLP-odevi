"""
Kendi el yazisi rakamlarini MNIST formatina cevirip egitilmis MLP ile tahmin eder.

MOD 1 - Tek tek dosyalar (my_digits/ klasoru, dosya adi: <sira>_<rakam>.png):
    python predict_own.py

MOD 2 - Tum numara tek gorselde (rakamlar birbirine degmeden, soldan saga):
    python predict_own.py --full numara.png --label 24120205005
"""
import argparse
import glob
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from train import MLP, HP, MEAN, STD

FOLDER = "my_digits"


# ---------------------------------------------------------------- on isleme
def clean(arr, thr=30):
    """Gri goruntu -> MNIST renk duzeni (siyah zemin, beyaz rakam), gurultu temizligi."""
    arr = arr.astype(np.float32)
    if arr.mean() > 127:            # acik zemin -> ters cevir
        arr = 255.0 - arr
    arr = np.clip(arr - np.median(arr), 0, None)   # zemin seviyesini sifirla
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    arr[arr < thr] = 0
    return arr


def to_mnist(arr):
    """Temizlenmis tek rakam goruntusu -> 28x28, [0,1]. Kirp, 20 px'e olcekle, kutle merkezine ortala."""
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        raise ValueError("Rakam bulunamadi")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.array(Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
                     dtype=np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    yy, xx = np.indices(canvas.shape)
    total = canvas.sum()
    cy, cx = (yy * canvas).sum() / total, (xx * canvas).sum() / total
    canvas = np.roll(canvas, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return canvas / 255.0


def segment_digits(arr):
    """Temizlenmis tam goruntuyu dikey bosluklara gore rakamlara boler (soldan saga)."""
    # 1) Yazi satirini bul: en yogun satir kumesini al, disindaki leke/kivrim gurultusunu at
    rows = (arr > 0).sum(axis=1)
    strong = np.where(rows >= 0.1 * rows.max())[0]
    row_gap = max(5, int(0.02 * arr.shape[0]))
    groups, cur = [], [strong[0]]
    for r in strong[1:]:
        if r - cur[-1] <= row_gap:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    best = max(groups, key=lambda g: rows[g[0]:g[-1] + 1].sum())
    pad = int(0.02 * arr.shape[0])
    arr = arr[max(0, best[0] - pad):best[-1] + pad + 1]

    # 2) Dikey bosluklara gore rakamlara bol
    cols = (arr > 0).sum(axis=0) > 0
    runs, start = [], None
    for i, v in enumerate(cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i]); start = None
    if start is not None:
        runs.append([start, len(cols)])

    # Birbirine cok yakin parcalari birlestir (ornegin 4 rakaminin ayri cizgileri)
    gap = max(2, int(0.01 * arr.shape[1]))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    # Cok kucuk lekeleri (gurultu) ele
    return [arr[:, a:b] for a, b in merged if (arr[:, a:b] > 0).sum() >= 30]


# ------------------------------------------------------------------ tahmin
def load_model():
    model = MLP(HP["hidden_sizes"])
    model.load_state_dict(torch.load("mlp_exp1.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model, img28):
    x = torch.tensor((img28 - MEAN) / STD).float().view(1, 1, 28, 28)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    return int(prob.argmax()), float(prob.max())


def report(imgs, preds, confs, truths, out_png):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.6))
    axes = np.atleast_1d(axes)
    for ax, im, p, c, t in zip(axes, imgs, preds, confs, truths):
        ax.imshow(im, cmap="gray"); ax.axis("off")
        ok = (t is None) or (p == t)
        title = f"Tahmin:{p}\n({c:.2f})" if t is None else f"Gercek:{t}\nTahmin:{p}\n({c:.2f})"
        ax.set_title(title, color="green" if ok else "red", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print("Tahmin edilen numara:", "".join(map(str, preds)))
    if all(t is not None for t in truths):
        print("Gercek numara       :", "".join(map(str, truths)))
        acc = np.mean(np.array(preds) == np.array(truths))
        print(f"Rakam bazli dogruluk: {acc:.2%}")
    print("Gorsel kaydedildi:", out_png)


def mode_files(model):
    files = sorted(
        glob.glob(os.path.join(FOLDER, "*.png")) + glob.glob(os.path.join(FOLDER, "*.jpg"))
        + glob.glob(os.path.join(FOLDER, "*.jpeg")),
        key=lambda p: int(os.path.basename(p).split("_")[0]))
    if not files:
        raise SystemExit(f"'{FOLDER}' klasorunde goruntu yok.")
    imgs, preds, confs, truths = [], [], [], []
    for f in files:
        img28 = to_mnist(clean(np.array(Image.open(f).convert("L"))))
        p, c = predict(model, img28)
        t = int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
        imgs.append(img28); preds.append(p); confs.append(c); truths.append(t)
        print(f"{os.path.basename(f)}: gercek={t} tahmin={p} guven={c:.3f}")
    report(imgs, preds, confs, truths, "own_digits_predictions.png")


def mode_full(model, path, label, save_crops=False):
    arr = clean(np.array(Image.open(path).convert("L")))
    parts = segment_digits(arr)
    print(f"{len(parts)} rakam bulundu.")
    if label and len(parts) != len(label):
        print(f"UYARI: Beklenen {len(label)} rakam, bulunan {len(parts)}. "
              "Rakamlar birbirine degiyor veya lekeler var olabilir; "
              "my_digits/segment_*.png dosyalarini kontrol et.")
    if save_crops and label and len(parts) == len(label):
        os.makedirs(FOLDER, exist_ok=True)
        for i, (p_arr, ch) in enumerate(zip(parts, label)):
            ys, xs = np.where(p_arr > 0)
            crop = p_arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop = np.pad(crop, 20)                      # kenarlara bosluk
            Image.fromarray((255 - crop).astype(np.uint8)).save(
                os.path.join(FOLDER, f"{i}_{ch}.png"))
        print(f"{len(parts)} rakam '{FOLDER}/' klasorune kaydedildi.")
    imgs, preds, confs = [], [], []
    for p_arr in parts:
        img28 = to_mnist(p_arr)
        p, c = predict(model, img28)
        imgs.append(img28); preds.append(p); confs.append(c)
    truths = [int(ch) for ch in label][:len(preds)] if label and len(parts) == len(label) \
        else [None] * len(preds)
    report(imgs, preds, confs, truths, "full_number_predictions.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", help="Tum numarayi iceren tek goruntu")
    ap.add_argument("--label", help="Gercek numara, ornegin 24120205005")
    ap.add_argument("--save_crops", action="store_true",
                    help="Bolunen rakamlari my_digits/ klasorune <sira>_<rakam>.png olarak kaydet")
    args = ap.parse_args()
    m = load_model()
    if args.full:
        mode_full(m, args.full, args.label, args.save_crops)
    else:
        mode_files(m)
