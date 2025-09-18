# app_sd_gui_cuda.py
# 單檔 GUI（tkinter），NVIDIA GPU（CUDA）用
# - 無參考圖：txt2img
# - 有參考圖：img2img
# - SD 1.5，fp16 on CUDA，較省 VRAM、相容性佳
# 建議：768x768 以上需要較多 VRAM；若顯存不夠，先用 512x512

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# --- 新增：統一 Hugging Face / Diffusers 快取位置（全機共用） ---
APP_NAME = "MyApp"  # 安裝檔名稱一致
PROGRAM_DATA = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
HF_CACHE = os.path.join(PROGRAM_DATA, APP_NAME, "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)

# 設定環境變數，確保 diffusers / huggingface 下載到這裡
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE
os.environ["DIFFUSERS_CACHE"] = HF_CACHE
# -------------------------------------------------------------

# -------------------- 基本設定 --------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 強制使用 NVIDIA CUDA（若不可用則報錯，因為這版專為 CUDA）
if not torch.cuda.is_available():
    raise RuntimeError("偵測不到 NVIDIA GPU（torch.cuda 不可用）。請確認已安裝 CUDA 版 PyTorch 與驅動。")

DEVICE = "cuda"
DTYPE = torch.float16  # 半精度，節省 VRAM

# 預先載入兩種 pipeline（txt2img / img2img）
print("Loading pipelines to CUDA…")
pipe_txt = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)
pipe_txt.enable_attention_slicing()

pipe_img = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)
pipe_img.enable_attention_slicing()

# -------------------- GUI --------------------
class SDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stable Diffusion GUI（CUDA / SD1.5）")
        self.geometry("1000x700")

        # 左側控制面板
        left = ttk.Frame(self, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        r = 0
        ttk.Label(left, text="提示詞 Prompt").grid(row=r, column=0, sticky="w")
        self.prompt = tk.Text(left, width=44, height=5)
        self.prompt.insert("1.0", "唐詩風山水：白日依山盡，黃河入海流，國畫筆觸、寫意留白、薄墨暈染")
        self.prompt.grid(row=r, column=1, pady=4, sticky="w")
        r += 1

        ttk.Label(left, text="負面詞 Negative").grid(row=r, column=0, sticky="w")
        self.neg = tk.Text(left, width=44, height=4)
        self.neg.insert("1.0", "low quality, blurry, artifacts, text, watermark")
        self.neg.grid(row=r, column=1, pady=4, sticky="w")
        r += 1

        ttk.Label(left, text="解析度 (W × H)").grid(row=r, column=0, sticky="w")
        row_wh = ttk.Frame(left)
        row_wh.grid(row=r, column=1, pady=2, sticky="w")
        self.w_var = tk.IntVar(value=512)
        self.h_var = tk.IntVar(value=512)
        ttk.Entry(row_wh, textvariable=self.w_var, width=7).pack(side=tk.LEFT)
        ttk.Label(row_wh, text=" × ").pack(side=tk.LEFT)
        ttk.Entry(row_wh, textvariable=self.h_var, width=7).pack(side=tk.LEFT)
        r += 1

        ttk.Label(left, text="步數 / CFG").grid(row=r, column=0, sticky="w")
        row_sc = ttk.Frame(left)
        row_sc.grid(row=r, column=1, pady=2, sticky="w")
        self.steps_var = tk.IntVar(value=28)     # 20~30 常用
        self.cfg_var = tk.DoubleVar(value=7.5)   # 7~8 常用
        ttk.Entry(row_sc, textvariable=self.steps_var, width=7).pack(side=tk.LEFT)
        ttk.Label(row_sc, text=" / ").pack(side=tk.LEFT)
        ttk.Entry(row_sc, textvariable=self.cfg_var, width=7).pack(side=tk.LEFT)
        r += 1

        ttk.Label(left, text="Seed（空=隨機）").grid(row=r, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="1234")
        ttk.Entry(left, textvariable=self.seed_var, width=12).grid(row=r, column=1, pady=2, sticky="w")
        r += 1

        ttk.Label(left, text="參考圖（img2img）").grid(row=r, column=0, sticky="w")
        row_img = ttk.Frame(left)
        row_img.grid(row=r, column=1, pady=2, sticky="w")
        self.img_path = tk.StringVar()
        ttk.Entry(row_img, textvariable=self.img_path, width=36).pack(side=tk.LEFT)
        ttk.Button(row_img, text="選檔", command=self.pick_image).pack(side=tk.LEFT, padx=6)
        r += 1

        ttk.Label(left, text="img2img 強度 strength（0.2~0.9）").grid(row=r, column=0, sticky="w")
        self.strength_var = tk.DoubleVar(value=0.6)
        ttk.Entry(left, textvariable=self.strength_var, width=7).grid(row=r, column=1, pady=2, sticky="w")
        r += 1

        row_btn = ttk.Frame(left)
        row_btn.grid(row=r, column=1, pady=8, sticky="w")
        ttk.Button(row_btn, text="生成", command=self.run_async).pack(side=tk.LEFT)
        ttk.Button(row_btn, text="另存圖片", command=self.save_image).pack(side=tk.LEFT, padx=8)
        r += 1

        self.status = tk.StringVar(value="狀態：待命（CUDA / fp16）")
        ttk.Label(left, textvariable=self.status, foreground="#444").grid(row=r, column=0, columnspan=2, sticky="w", pady=8)

        # 右側預覽面板
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.preview = tk.Label(right, bd=1, relief=tk.SUNKEN)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self._current_pil = None

        # 窗口調整時自動重繪預覽
        self.preview.bind("<Configure>", lambda e: self._refresh_preview())

    def pick_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp")]
        )
        if path:
            self.img_path.set(path)

    def run_async(self):
        threading.Thread(target=self.generate, daemon=True).start()

    def generate(self):
        try:
            prompt = self.prompt.get("1.0", "end").strip()
            negative = self.neg.get("1.0", "end").strip() or None
            W, H = int(self.w_var.get()), int(self.h_var.get())
            steps = max(5, int(self.steps_var.get()))
            cfg = float(self.cfg_var.get())
            strength = float(self.strength_var.get())
            seed_txt = self.seed_var.get().strip()
            seed = None if seed_txt == "" else int(seed_txt)

            # 生成器（可重現）
            g = None
            if seed is not None:
                g = torch.Generator(device=DEVICE).manual_seed(seed)

            ref_path = self.img_path.get().strip()
            use_img2img = os.path.isfile(ref_path)

            self.status.set(f"狀態：推論中…（{'img2img' if use_img2img else 'txt2img'}）")
            self.update_idletasks()

            if use_img2img:
                # 以圖生圖
                init = Image.open(ref_path).convert("RGB")
                init = init.resize((W, H))  # 可依需求改為保持比例後 pad
                out = pipe_img(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=init,
                    strength=max(0.2, min(0.95, strength)),
                    num_inference_steps=steps,
                    guidance_scale=max(1.0, cfg),
                    generator=g
                )
            else:
                # 文字生圖
                out = pipe_txt(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=W, height=H,
                    num_inference_steps=steps,
                    guidance_scale=max(1.0, cfg),
                    generator=g
                )

            img = out.images[0]
            self._set_preview(img)

            # 自動存一份
            out_path = os.path.join(OUT_DIR, "result.png")
            img.save(out_path)
            self.status.set(f"狀態：完成 ✓ 已儲存 {out_path}")

        except Exception as e:
            self.status.set("狀態：發生錯誤")
            messagebox.showerror("錯誤", str(e))

    def _set_preview(self, pil_img):
        self._current_pil = pil_img
        self._refresh_preview()

    def _refresh_preview(self):
        if self._current_pil is None:
            return
        w = self.preview.winfo_width() or 640
        h = self.preview.winfo_height() or 640
        show = self._current_pil.copy()
        show.thumbnail((w, h))
        tk_img = ImageTk.PhotoImage(show)
        self.preview.configure(image=tk_img)
        self.preview.image = tk_img  # 避免被 GC

    def save_image(self):
        if self._current_pil is None:
            messagebox.showinfo("提示", "尚未生成圖片")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("WEBP", "*.webp")]
        )
        if path:
            self._current_pil.save(path)
            self.status.set(f"狀態：已另存 {path}")

if __name__ == "__main__":
    app = SDApp()
    app.mainloop()
