"""
scripts/build_celebrity_index.py

Script chay OFFLINE, MOT LAN DUY NHAT (tren may co ket noi Internet), de
xay dung file du lieu nguoi noi tieng Viet Nam dung cho tinh nang nhan
dien nguoi noi tieng (yeu cau 1.3.1 trong tai lieu yeu cau).

KHONG chay script nay tren server production - no chi dung de TAO SAN file
`app/data/celebrity_embeddings.npz`, sau do commit file .npz nay (chi vai
MB) vao source code. Luc chay that, `app/services/celebrity_detection.py`
chi doc file .npz co san, khong can tai dataset hay ket noi Internet.

## Dataset su dung
fptudsc/face-celeb-vietnamese (HuggingFace Datasets)
https://huggingface.co/datasets/fptudsc/face-celeb-vietnamese
- Giay phep: Apache-2.0
- ~8.557 anh cua 224 nguoi noi tieng Viet Nam (ca si, dien vien, hoa hau)
- Nguon: thu thap tu website nguoinoitieng.tv
- Cot du lieu: `image` (anh), `label` (ten nguoi noi tieng, vi du
  "ca sĩ Akira Phan")

## Cach hoat dong
1. Tai dataset ve tu HuggingFace Hub (dung thu vien `datasets`).
2. Voi moi anh: phat hien khuon mat bang insightface, lay khuon mat lon
   nhat, trich xuat vector dac trung (embedding) 512 chieu (ArcFace).
   Bo qua anh khong phat hien duoc khuon mat ro rang (anh loi, qua nho,
   nhieu nguoi mo, v.v.) - dataset scrape tu web nen khong tranh khoi mot
   ti le nhiễu nhat dinh.
3. Gom tat ca embedding theo tung nguoi (tung nhan `label`), lay trung
   binh roi chuan hoa lai (L2-normalize) -> moi nguoi noi tieng chi con 1
   vector dai dien 512 chieu.
4. Luu ket qua vao `app/data/celebrity_embeddings.npz` gom 2 mang:
     - names: mang ten nguoi noi tieng, shape (N,)
     - embeddings: mang vector dai dien, shape (N, 512), dtype float32

## Cai dat truoc khi chay (KHONG can trong requirements.txt cua server,
## day la dependency chi dung 1 lan luc build du lieu):
    pip install datasets insightface onnxruntime pillow numpy huggingface_hub

## Chay:
    python scripts/build_celebrity_index.py

Thoi gian chay: tuy CPU, thuong 15-40 phut cho ~8.5k anh tren CPU thong
thuong (khong can GPU, nhung co GPU se nhanh hon nhieu - xem bien
CTX_ID ben duoi de bat GPU).
"""
import os
import sys
from collections import defaultdict

import numpy as np

# Doi ctx_id sang 0 (hoac id GPU tuong ung) neu may co GPU + da cai
# onnxruntime-gpu, se chay nhanh hon dang ke. Mac dinh -1 = chay tren CPU.
CTX_ID = -1

# Model insightface: 'buffalo_l' (chinh xac hon, mac dinh) hoac 'buffalo_s'
# (nhe/nhanh hon, phu hop may cau hinh thap). PHAI KHOP voi
# CELEBRITY_MODEL_NAME dang dung trong settings.py / celebrity_detection.py.
MODEL_NAME = os.environ.get('CELEBRITY_MODEL_NAME', 'buffalo_l')

DATASET_NAME = 'fptudsc/face-celeb-vietnamese'

# Bo qua khuon mat nho hon MIN_FACE_SIZE pixel (canh ngan nhat cua bbox) -
# thuong la nguoi phia sau/mo, khong phai chu the chinh cua anh.
MIN_FACE_SIZE = 40

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'app', 'data', 'celebrity_embeddings.npz')


def log(msg):
    print(f'[build_celebrity_index] {msg}', flush=True)


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Thieu thu vien 'datasets'. Cai bang: pip install datasets huggingface_hub"
        )
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        sys.exit(
            "Thieu thu vien 'insightface'. Cai bang: pip install insightface onnxruntime"
        )
    try:
        import cv2
    except ImportError:
        sys.exit(
            "Thieu thu vien 'opencv-python' (di kem insightface). Cai bang: pip install opencv-python"
        )

    log(f"Dang tai dataset '{DATASET_NAME}' tu HuggingFace Hub...")
    ds = load_dataset(DATASET_NAME, split='train')
    total = len(ds)
    log(f'Tong so anh trong dataset: {total}')

    log(f"Dang khoi tao model nhan dien khuon mat insightface ('{MODEL_NAME}')...")
    # allowed_modules=['detection', 'recognition']: buffalo_l mac dinh load
    # ca 5 model (detection, recognition, landmark_3d_68, landmark_2d_106,
    # genderage). Ta chi can detection (tim khuon mat) + recognition (trich
    # embedding ArcFace) de so sanh, nen bo qua 3 model con lai giup nhanh
    # hon dang ke (~2-3 lan) ma khong anh huong ket qua.
    face_app = FaceAnalysis(
        name=MODEL_NAME,
        providers=['CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition'],
    )
    face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))

    sums = defaultdict(lambda: np.zeros(512, dtype=np.float64))
    counts = defaultdict(int)
    skipped_no_face = 0
    skipped_bad_row = 0

    for i, row in enumerate(ds):
        name = row.get('label')
        img = row.get('image')

        if img is None or not name:
            skipped_bad_row += 1
            continue

        try:
            img_rgb = np.array(img.convert('RGB'))
        except Exception as e:
            log(f'  Loi doc anh dong {i}: {e}')
            skipped_bad_row += 1
            continue

        img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])  # RGB -> BGR cho insightface
        # QUAN TRONG: bat buoc phai ascontiguousarray() sau khi dao truc mau
        # bang [:, :, ::-1] - phep dao nay tao ra 1 "view" khong lien tuc
        # trong bo nho (non-contiguous), khien cv2.resize() ben trong
        # insightface doc sai du lieu anh (khong loi, nhung tra ve 0 khuon
        # mat cho hau het anh). Day la nguyen nhan gay loi "bo qua gan het
        # dataset" da gap truoc do.

        try:
            faces = face_app.get(img_bgr)
        except Exception as e:
            log(f'  Loi phat hien khuon mat dong {i}: {e}')
            skipped_no_face += 1
            continue

        if not faces:
            # Mot so anh trong dataset co the la anh crop rat sat vao mat
            # (khong con vien/context xung quanh) khien SCRFD kho nhan ra
            # o do phan giai 640x640 mac dinh. Thu lai voi vien them
            # (padding) va nguong nhay hon truoc khi bo qua han.
            h, w = img_bgr.shape[:2]
            pad_h, pad_w = int(h * 0.4), int(w * 0.4)
            padded_bgr = cv2.copyMakeBorder(
                img_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101
            )
            try:
                faces = face_app.get(padded_bgr)
            except Exception:
                faces = []

        if not faces:
            skipped_no_face += 1
            continue

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            skipped_no_face += 1
            continue

        sums[name] += face.normed_embedding
        counts[name] += 1

        if (i + 1) % 200 == 0 or (i + 1) == total:
            log(f'  ...da xu ly {i + 1}/{total} anh | '
                f'{len(sums)} nguoi co du lieu | '
                f'bo qua (khong phat hien mat) {skipped_no_face} | '
                f'bo qua (loi/thieu du lieu) {skipped_bad_row}')

    if not sums:
        sys.exit('Khong trich xuat duoc embedding nao. Kiem tra lai dataset/model.')

    names = []
    embeddings = []
    for name, total_vec in sums.items():
        mean_emb = total_vec / counts[name]
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        names.append(name)
        embeddings.append(mean_emb.astype(np.float32))

    names_arr = np.array(names)
    embeddings_arr = np.stack(embeddings).astype(np.float32)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, names=names_arr, embeddings=embeddings_arr)

    log('')
    log('=' * 70)
    log(f'HOAN TAT: {len(names_arr)} nguoi noi tieng, luu tai:')
    log(f'  {OUTPUT_PATH}')
    log(f'Tong anh bo qua: {skipped_no_face} (khong ro mat) + '
        f'{skipped_bad_row} (loi/thieu du lieu) / {total} anh goc')
    log('Nho commit file .npz nay vao git de server production su dung.')
    log('=' * 70)


if __name__ == '__main__':
    main()