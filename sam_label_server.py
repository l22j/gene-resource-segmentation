"""
AI 기반 이미지 세분화 시스템 - 파이썬 서버 버전 (로컬 서버)

실행:
    C:\\Users\\Deep_Learning\\anaconda3\\python.exe sam_label_server.py --images <이미지폴더>
브라우저:
    http://127.0.0.1:8765

추가 패키지 설치 없이 표준 라이브러리 http.server 로 동작합니다.
"""
import argparse
import base64
import io
import json
import os
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image, ImageOps
from segment_anything import SamPredictor, sam_model_registry

ROOT = os.path.dirname(os.path.abspath(__file__))

# key -> (모델 타입, 가중치 파일명, 표시 이름)
WEIGHTS = OrderedDict([
    ("vit_b", ("vit_b", "sam_vit_b_01ec64.pth", "SAM ViT-B (빠름 · 375MB)")),
    ("vit_l", ("vit_l", "sam_vit_l_0b3195.pth", "SAM ViT-L (중간 · 1.2GB)")),
    ("vit_h", ("vit_h", "sam_vit_h_4b8939.pth", "SAM ViT-H (정확 · 2.5GB)")),
])
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EMBED_CACHE_SIZE = 24

CFG = {"image_dir": "", "output_dir": "", "custom_output": "", "weight_dir": ROOT}


def documents_dir():
    """윈도우 '문서' 폴더 (OneDrive 로 옮겨진 경우도 반영)"""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
            return buf.value
    except Exception:  # noqa
        pass
    return os.path.join(os.path.expanduser("~"), "Documents")


# 배포 기준 기본 위치: 프로그램 폴더(쓰기 금지일 수 있음)나 이미지 폴더(남의 데이터셋)에 쓰지 않음
APP_HOME = os.path.join(documents_dir(), "SAM_Labeler")                       # 이미지 · 결과 기본 위치
CONFIG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "SAM_Labeler")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")
LEGACY_SETTINGS = os.path.join(ROOT, "sam_label_settings.json")
SOURCE_MARK = "_source_folder.txt"   # 결과 폴더가 어느 이미지 폴더 것인지 기록


def is_writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        t = os.path.join(d, ".write_test")
        with open(t, "w") as f:
            f.write("")
        os.remove(t)
        return True
    except OSError:
        return False


def load_settings():
    for p in (SETTINGS_PATH, LEGACY_SETTINGS):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def save_settings():
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"image_dir": CFG["image_dir"], "custom_output": CFG["custom_output"]},
                      f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[SAM] 설정 저장 실패 (무시하고 계속): {e}")


def default_image_dir():
    d = os.path.join(ROOT, "images")
    return d if is_writable(d) else os.path.join(APP_HOME, "images")


def default_output_dir(image_dir):
    """
    기본 저장 위치: 문서\SAM_Labeler\labels\<이미지폴더 이름>
    - 예전 버전처럼 <이미지폴더>\sam_labels 에 이미 결과가 있으면 그대로 이어서 사용
    - 이름이 같은 다른 이미지 폴더와 섞이지 않도록 _2, _3 … 을 붙임
    """
    legacy = os.path.join(image_dir, "sam_labels")
    if os.path.isdir(os.path.join(legacy, "annotations")):
        return legacy
    base = os.path.join(APP_HOME, "labels")
    name = os.path.basename(os.path.normpath(image_dir)) or "images"
    for i in range(1, 1000):
        cand = os.path.join(base, name if i == 1 else f"{name}_{i}")
        if not os.path.exists(cand):
            return cand
        try:
            with open(os.path.join(cand, SOURCE_MARK), encoding="utf-8") as f:
                if os.path.normcase(f.read().strip()) == os.path.normcase(image_dir):
                    return cand
        except OSError:
            pass
    return os.path.join(base, name)


def update_output_dir():
    CFG["output_dir"] = CFG["custom_output"] or default_output_dir(CFG["image_dir"])


def set_image_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise ValueError(f"폴더가 없습니다: {path}")
    CFG["image_dir"] = path
    update_output_dir()


def set_output_dir(path):
    """빈 문자열이면 기본 위치로 되돌림"""
    path = (path or "").strip()
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not is_writable(path):
            raise ValueError(f"이 폴더에는 저장할 수 없습니다 (권한 없음/읽기 전용): {path}")
    CFG["custom_output"] = path
    update_output_dir()


def pick_folder(title, initial):
    """서버 PC에 윈도우 폴더 선택 창을 띄움 (취소하면 빈 문자열)"""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(parent=root, title=title,
                                       initialdir=initial if os.path.isdir(initial or "") else None)
    finally:
        root.destroy()
    return os.path.normpath(path) if path else ""


def list_images():
    d = CFG["image_dir"]
    names = [f for f in os.listdir(d)
             if os.path.splitext(f)[1].lower() in IMG_EXT and os.path.isfile(os.path.join(d, f))]
    names.sort()
    return names


def image_path(name):
    # 경로 조작 방지: 이미지 폴더 바로 아래 파일만 허용
    if not name or os.path.basename(name) != name:
        raise ValueError("잘못된 파일 이름")
    p = os.path.join(CFG["image_dir"], name)
    if not os.path.isfile(p):
        raise FileNotFoundError(name)
    return p


def load_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # 브라우저 표시 방향과 좌표를 맞춤
    return img.convert("RGB")


def stem_of(name):
    return os.path.splitext(name)[0]


def annotation_path(name):
    return os.path.join(CFG["output_dir"], "annotations", stem_of(name) + ".json")


def mask_to_b64(mask):
    """bool HxW -> 흰색 + 알파(마스크) PNG base64"""
    a = mask.astype(np.uint8) * 255
    la = np.stack([np.full_like(a, 255), a], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(la, mode="LA").save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_mask(b64):
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    if "A" in img.getbands():
        return np.array(img.getchannel("A")) > 127
    return np.array(img.convert("L")) > 127


class Engine:
    def __init__(self, device):
        self.device = device
        self.lock = threading.RLock()
        self.key = None
        self.predictor = None
        self.cache = OrderedDict()  # (weight, name, mtime) -> (features, original_size, input_size)
        self.current = None
        self.status = "모델 없음"

    def load(self, key):
        if key not in WEIGHTS:
            raise ValueError(f"알 수 없는 가중치: {key}")
        with self.lock:
            if self.key == key:
                return 0.0
            t = time.time()
            self.status = f"{key} 로딩 중"
            self.predictor = None
            self.key = None
            self.cache.clear()
            self.current = None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            mtype, fname, _ = WEIGHTS[key]
            ckpt = os.path.join(CFG["weight_dir"], fname)
            sam = sam_model_registry[mtype](checkpoint=ckpt)
            sam.to(self.device).eval()
            self.predictor = SamPredictor(sam)
            self.key = key
            self.status = f"{key} 준비됨"
            return time.time() - t

    def ensure_image(self, name):
        if self.predictor is None:
            raise RuntimeError("가중치를 먼저 선택하세요")
        path = image_path(name)
        ck = (self.key, name, os.path.getmtime(path))
        if self.current == ck:
            return False
        p = self.predictor
        if ck in self.cache:
            p.features, p.original_size, p.input_size = self.cache[ck]
            p.is_image_set = True
            self.cache.move_to_end(ck)
        else:
            with torch.inference_mode():
                p.set_image(np.array(load_rgb(path)))
            self.cache[ck] = (p.features, p.original_size, p.input_size)
            while len(self.cache) > EMBED_CACHE_SIZE:
                self.cache.popitem(last=False)
        self.current = ck
        return True

    def embed(self, name):
        with self.lock:
            t = time.time()
            computed = self.ensure_image(name)
            return computed, time.time() - t

    def predict(self, name, points, labels, multimask):
        with self.lock:
            self.ensure_image(name)
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.asarray(points, dtype=np.float32),
                    point_labels=np.asarray(labels, dtype=np.int32),
                    multimask_output=multimask,
                )
            return masks, scores


ENGINE = None
PICK_LOCK = threading.Lock()

PALETTE = [(46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (230, 126, 34),
           (26, 188, 156), (231, 76, 60), (236, 112, 190), (149, 165, 166), (39, 174, 96)]


def save_annotation(body):
    name = body["name"]
    img = load_rgb(image_path(name))
    W, H = img.size
    out = CFG["output_dir"]
    try:
        for sub in ("annotations", "masks", "instances", "overlays"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        mark = os.path.join(out, SOURCE_MARK)
        if not os.path.exists(mark):
            with open(mark, "w", encoding="utf-8") as f:
                f.write(CFG["image_dir"])
    except OSError as e:
        raise ValueError(f"저장 폴더에 쓸 수 없습니다. 왼쪽 '저장 폴더'에서 다른 폴더를 지정하세요. ({e})")

    union = np.zeros((H, W), bool)
    inst = np.zeros((H, W), np.uint16)
    overlay = np.array(img, dtype=np.float32)
    objs_meta = []
    for i, o in enumerate(body.get("objects", [])):
        meta = {k: o.get(k) for k in ("name", "points", "strokes", "mask_index", "user_pick", "score", "color")}
        meta["id"] = i + 1
        m = None
        if o.get("mask_png"):
            m = b64_to_mask(o["mask_png"])
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
        if m is not None and m.any():
            union |= m
            inst[m] = i + 1
            col = np.array(PALETTE[i % len(PALETTE)], np.float32)
            overlay[m] = overlay[m] * 0.45 + col * 0.55
            meta["area"] = int(m.sum())
            ys, xs = np.nonzero(m)
            meta["bbox_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            meta["area"] = 0
        objs_meta.append(meta)

    stem = stem_of(name)
    Image.fromarray(union.astype(np.uint8) * 255).save(os.path.join(out, "masks", stem + ".png"))
    inst_img = Image.fromarray(inst.astype(np.uint8) if inst.max() < 256 else inst)
    inst_img.save(os.path.join(out, "instances", stem + ".png"))
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(
        os.path.join(out, "overlays", stem + ".jpg"), quality=92)

    ann = {
        "image": name,
        "width": W,
        "height": H,
        "weight": body.get("weight"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_label": {"1": "작물(전경)", "0": "배경"},
        "objects": objs_meta,
    }
    with open(annotation_path(name), "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    return {"ok": True, "dir": out, "area": int(union.sum()), "ratio": float(union.mean())}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        # html 파일을 직접 연 경우(Origin: null)와 로컬 주소만 허용
        origin = self.headers.get("Origin")
        if origin and (origin == "null" or urlparse(origin).hostname in ("127.0.0.1", "localhost")):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _safe(self, fn):
        try:
            fn()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._send(500, {"error": "GPU 메모리 부족 (다른 프로그램의 GPU 사용을 줄이거나 더 작은 가중치를 선택하세요)"})
        except FileNotFoundError as e:
            self._send(404, {"error": f"파일 없음: {e}"})
        except Exception as e:  # noqa
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def do_GET(self):
        self._safe(self._get)

    def do_POST(self):
        self._safe(self._post)

    def _get(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "sam_label.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path == "/api/info":
            weights = [{"key": k, "label": v[2],
                        "available": os.path.isfile(os.path.join(CFG["weight_dir"], v[1]))}
                       for k, v in WEIGHTS.items()]
            self._send(200, {"weights": weights, "current": ENGINE.key, "device": ENGINE.device,
                             "image_dir": CFG["image_dir"], "output_dir": CFG["output_dir"],
                             "custom_output": bool(CFG["custom_output"])})
        elif u.path == "/api/images":
            names = list_images()
            done = [os.path.isfile(annotation_path(n)) for n in names]
            self._send(200, {"images": [{"name": n, "done": d} for n, d in zip(names, done)],
                             "image_dir": CFG["image_dir"], "output_dir": CFG["output_dir"],
                             "custom_output": bool(CFG["custom_output"])})
        elif u.path == "/api/image":
            img = load_rgb(image_path(q["name"]))
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=1) if q.get("lossless") else img.save(buf, format="JPEG", quality=95)
            self._send(200, buf.getvalue(), "image/png" if q.get("lossless") else "image/jpeg")
        elif u.path == "/api/annotation":
            p = annotation_path(q["name"])
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    self._send(200, json.load(f))
            else:
                self._send(200, {})
        else:
            self._send(404, {"error": "not found"})

    def _post(self):
        path = urlparse(self.path).path
        body = self._json()
        if path == "/api/load_weight":
            sec = ENGINE.load(body["key"])
            self._send(200, {"ok": True, "key": ENGINE.key, "sec": sec})
        elif path == "/api/set_dir":
            set_image_dir(body["path"])
            save_settings()
            self._send(200, {"ok": True, "image_dir": CFG["image_dir"], "output_dir": CFG["output_dir"]})
        elif path == "/api/set_output":
            set_output_dir(body.get("path", ""))
            save_settings()
            self._send(200, {"ok": True, "output_dir": CFG["output_dir"], "custom": bool(CFG["custom_output"])})
        elif path == "/api/pick_dir":
            kind = body.get("kind")
            with PICK_LOCK:
                p = pick_folder("저장할 폴더 선택" if kind == "output" else "이미지 폴더 선택",
                                CFG["output_dir"] if kind == "output" else CFG["image_dir"])
            self._send(200, {"path": p})
        elif path == "/api/embed":
            computed, sec = ENGINE.embed(body["name"])
            self._send(200, {"ok": True, "computed": computed, "sec": sec})
        elif path == "/api/predict":
            t = time.time()
            masks, scores = ENGINE.predict(body["name"], body["points"], body["labels"],
                                           bool(body.get("multimask")))
            self._send(200, {"masks": [mask_to_b64(m) for m in masks],
                             "scores": [float(s) for s in scores], "sec": time.time() - t})
        elif path == "/api/save":
            self._send(200, save_annotation(body))
        elif path == "/api/upload":
            fname = os.path.basename(body["filename"])
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                raise ValueError("이미지 파일만 업로드할 수 있습니다")
            data = body["data"].split(",", 1)[-1]
            with open(os.path.join(CFG["image_dir"], fname), "wb") as f:
                f.write(base64.b64decode(data))
            self._send(200, {"ok": True, "name": fname})
        else:
            self._send(404, {"error": "not found"})


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="라벨링할 이미지 폴더 (기본: 마지막에 쓴 폴더)")
    ap.add_argument("--output", default=None, help="결과 저장 폴더 (기본: 문서\SAM_Labeler\labels\<이미지폴더 이름>)")
    ap.add_argument("--weights", default=ROOT, help="SAM 가중치(.pth) 폴더")
    ap.add_argument("--weight", default="vit_b", choices=list(WEIGHTS), help="시작 시 불러올 가중치")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="준비되면 브라우저 자동 열기")
    ap.add_argument("--settings", default=None, help="설정 파일 경로 (기본: %%APPDATA%%\SAM_Labeler\settings.json)")
    args = ap.parse_args()

    # 윈도우에서 같은 포트에 서버가 두 개 뜨지 않도록 주소 재사용을 끄고, 모델 로딩 전에 포트부터 확인
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        print(f"[SAM] 포트 {args.port} 가 이미 사용 중입니다. 라벨러가 이미 실행 중이면 그 창을 쓰거나, 닫고 다시 실행하세요.")
        return

    global SETTINGS_PATH
    if args.settings:
        SETTINGS_PATH = os.path.abspath(args.settings)
    st = load_settings()
    images = args.images or st.get("image_dir") or default_image_dir()
    if not os.path.isdir(images):
        print(f"[SAM] 이미지 폴더가 없어 기본 폴더를 사용합니다: {images}")
        images = default_image_dir()
    os.makedirs(images, exist_ok=True)
    set_image_dir(images)
    try:
        set_output_dir(args.output if args.output is not None else st.get("custom_output", ""))
    except ValueError as e:
        print(f"[SAM] {e} → 기본 저장 폴더를 사용합니다")
        set_output_dir("")
    CFG["weight_dir"] = os.path.abspath(args.weights)
    ENGINE = Engine(args.device)
    print(f"[SAM] device={args.device}  weight={args.weight} 로딩 중...")
    print(f"[SAM] {ENGINE.load(args.weight):.1f}s")
    print(f"[SAM] 이미지 폴더: {CFG['image_dir']}")
    print(f"[SAM] 저장 폴더: {CFG['output_dir']}")
    print(f"[SAM] 브라우저에서 열기: http://{args.host}:{args.port}")
    if args.open:
        import webbrowser
        webbrowser.open(f"http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
