# Stable Diffusion GUI (CUDA + Tkinter)

This project provides a lightweight **desktop GUI** for **Stable Diffusion 1.5**, built with **Tkinter** and optimized for **NVIDIA GPUs (CUDA, FP16)**.  
It allows users to generate images via **text-to-image** (txt2img) and **image-to-image** (img2img) pipelines using the [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library.

![GUI Preview](docs/screenshot.png) <!-- optional: add a screenshot if available -->

---

## ✨ Features
- **CUDA acceleration**: optimized with `torch.float16` for reduced VRAM usage.  
- **Two pipelines**: 
  - Text-to-Image (txt2img)  
  - Image-to-Image (img2img, with adjustable strength)  
- **Customizable parameters**:
  - Prompt & negative prompt  
  - Image resolution (width × height)  
  - Number of inference steps  
  - Guidance scale (CFG)  
  - Seed (for reproducibility)  
- **Tkinter GUI**:
  - Simple control panel for all parameters  
  - Live preview of generated images  
  - One-click save/export  

---

## 🖥 Development Environment
- Developed with **[Vibe Coding](https://vibe.dev/)**  
- Python 3.10+  
- NVIDIA GPU with CUDA 11.8+  

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
````

---

## 🚀 Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/stable-diffusion-gui.git
   cd stable-diffusion-gui
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the GUI:

   ```bash
   python StableDiffusionGUI.py
   ```

4. Enter your prompt, adjust settings, and click **Generate**.
   The output will be automatically saved in the `outputs/` directory.

---

## 📂 Project Structure

```
.
├── StableDiffusionGUI.py   # Main Tkinter-based GUI application
├── requirements.txt        # Dependency list
├── outputs/                # Generated results (auto-created)
└── docs/                   # Documentation, screenshots (optional)
```

---

## ⚖️ License & Usage Policy

This project is released under the **MIT License**.

However, please note:

* It relies on **third-party models and libraries** such as [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), and [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), which are subject to the **Apache 2.0 license** and their respective **model usage terms**.
* Users must comply with the **[Hugging Face Terms of Service](https://huggingface.co/terms-of-service)** and the **[Stable Diffusion Model License](https://huggingface.co/runwayml/stable-diffusion-v1-5)** when generating or distributing images.
* This project is intended **for research and educational purposes only**.

---

## 🙌 Acknowledgements

* [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
* [PyTorch](https://pytorch.org/)
* [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

---

## 🔬 Author

Developed by **Yu-Tse Wu (吳雨澤)**
* GitHub: [@Yu-Tse](https://github.com/Yu-Tse)


