"""SAM 가중치(.pth) 다운로드 도구 - 파이썬 서버 버전용.

받은 파일은 sam_label_server.py 와 같은 폴더에 저장되며, 서버가 자동으로 인식합니다.
파이썬 기본 라이브러리만 사용하므로 별도 설치가 필요 없습니다.

    python download_weights.py            # 대화형으로 선택 (여러 개 가능)
    python download_weights.py b l        # ViT-B, ViT-L 받기
    python download_weights.py all        # 전부 받기
    python download_weights.py --list     # 받은 상태만 확인
"""
import argparse
import os
import sys
import time
import urllib.request
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = "https://dl.fbaipublicfiles.com/segment_anything/"

# 키 -> (파일명, 표시 이름, 대략 크기 MB)
WEIGHTS = OrderedDict([
    ("b", ("sam_vit_b_01ec64.pth", "ViT-B (빠름 · 가장 작음)", 375)),
    ("l", ("sam_vit_l_0b3195.pth", "ViT-L (중간 · 권장)", 1250)),
    ("h", ("sam_vit_h_4b8939.pth", "ViT-H (가장 정확 · 가장 큼)", 2560)),
])


def exists(key):
    return os.path.isfile(os.path.join(ROOT, WEIGHTS[key][0]))


def show():
    print("\n  번호  모델                            크기      상태")
    for i, (k, (fn, label, mb)) in enumerate(WEIGHTS.items(), 1):
        st = "받음" if exists(k) else "없음"
        print(f"  {i}     {label:<28} {mb / 1000:>5.2f} GB   {st}")
    print()


def ask():
    show()
    s = input("받을 번호를 쉼표/공백으로 입력 (예: 1,3 · all=전부 · 엔터=취소): ").strip().lower()
    if not s:
        return []
    if s in ("all", "a", "전부"):
        return list(WEIGHTS)
    keys = list(WEIGHTS)
    out = []
    for t in s.replace(",", " ").split():
        if t.isdigit() and 1 <= int(t) <= len(keys):
            out.append(keys[int(t) - 1])
        elif t in WEIGHTS:
            out.append(t)
        else:
            sys.exit(f"알 수 없는 입력: {t}")
    return list(dict.fromkeys(out))


def download(key):
    fn, label, mb = WEIGHTS[key]
    dest = os.path.join(ROOT, fn)
    part = dest + ".part"
    if os.path.isfile(dest):
        print(f"[건너뜀] {fn} 이미 있음")
        return True
    done = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(BASE + fn)
    if done:
        req.add_header("Range", f"bytes={done}-")
    print(f"[받기] {label} -> {fn}" + (f" (이어받기 {done / 1e6:.0f}MB 부터)" if done else ""))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if done and r.status != 206:  # 서버가 이어받기를 안 해주면 처음부터
                done = 0
            total = done + int(r.headers.get("Content-Length", 0))
            t0, got = time.time(), 0
            with open(part, "ab" if done else "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    cur = done + got
                    sp = got / max(time.time() - t0, 1e-6) / 1e6
                    pct = f"{cur / total * 100:5.1f}%" if total else ""
                    print(f"\r  {cur / 1e6:8.1f} / {total / 1e6:.1f} MB  {pct}  {sp:.1f} MB/s", end="", flush=True)
        print()
        if total and os.path.getsize(part) != total:
            print(f"[실패] {fn} 크기가 맞지 않습니다. 다시 실행하면 이어받습니다.")
            return False
        os.replace(part, dest)
        print(f"[완료] {fn}")
        return True
    except Exception as e:
        print(f"\n[실패] {fn}: {e}\n  다시 실행하면 이어받습니다.")
        return False


def main():
    ap = argparse.ArgumentParser(description="SAM 가중치 다운로드")
    ap.add_argument("models", nargs="*", help="b l h all (여러 개 가능, 비우면 대화형)")
    ap.add_argument("--list", action="store_true", help="받은 상태만 보기")
    a = ap.parse_args()
    if a.list:
        show()
        return
    if a.models:
        keys = list(WEIGHTS) if "all" in a.models else [m.lower().replace("vit_", "") for m in a.models]
        bad = [k for k in keys if k not in WEIGHTS]
        if bad:
            sys.exit(f"알 수 없는 모델: {', '.join(bad)} (b, l, h, all 중 선택)")
        keys = list(dict.fromkeys(keys))
    else:
        keys = ask()
    if not keys:
        print("받을 모델이 없습니다.")
        return
    ok = [download(k) for k in keys]
    print("\n모두 완료. run_sam_label.bat 을 실행하면 자동으로 인식됩니다." if all(ok) else "\n일부 실패했습니다.")


if __name__ == "__main__":
    main()
