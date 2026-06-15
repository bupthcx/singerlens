# -*- coding: utf-8 -*-
"""驱动本地 Chrome 上传真实音频到 SingerLens Gradio(localhost:7860)并截取带结果的界面。"""
import sys, time, pathlib
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
clip = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "singer_a_dearfriend_002_fake.wav")
out = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "demo_shot.png")
clip = str(pathlib.Path(clip).resolve())

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu"])
    page = browser.new_page(viewport={"width": 1320, "height": 1700}, device_scale_factor=2)
    page.goto("http://localhost:7860/", wait_until="networkidle", timeout=60000)
    page.wait_for_selector("text=SingerLens", timeout=30000)

    # 上传音频:定位隐藏的 file input
    finp = page.locator('input[type="file"]').first
    finp.wait_for(state="attached", timeout=30000)
    finp.set_input_files(clip)
    print("uploaded:", clip, flush=True)
    time.sleep(3)  # 等 gradio 接收文件

    # 点击“分析”
    page.get_by_role("button", name="分析").click()
    print("clicked 分析, waiting for result...", flush=True)

    # 等判定结果出现(verdict markdown 含“真实性评分”)
    page.wait_for_selector("text=真实性评分", timeout=180000)
    print("verdict rendered", flush=True)
    # 等情感面板(可能含“情感”或一致性数字),给服务端 Whisper 时间;失败不致命
    try:
        page.wait_for_selector("text=一致", timeout=120000)
        print("emotion rendered", flush=True)
    except Exception as e:
        print("emotion wait skipped:", repr(e), flush=True)
    time.sleep=getattr(time, "sleep")
    time.sleep(4)  # 等雷达/时间轴图片绘制完成

    # 打印结果文本核对
    try:
        body = page.inner_text("body")
        for kw in ("判断", "真实性评分", "AI 翻唱概率"):
            idx = body.find(kw)
            if idx >= 0:
                print("[txt]", body[idx:idx+40].replace("\n", " "), flush=True)
    except Exception:
        pass

    page.screenshot(path=out, full_page=True)
    print("SHOT:", out, flush=True)
    browser.close()
