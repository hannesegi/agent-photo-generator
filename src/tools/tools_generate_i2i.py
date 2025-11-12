import base64, json, requests, sys, os, time
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional


class SDImg2Img:
    ENDPOINT = "http://172.16.100.249:7861/sdapi/v1/img2img"

    # ---------- helper baca gambar -> base64 ----------
    @staticmethod
    def file_to_base64(path: str) -> str:
        """Mengambil file apapun, konversi ke PNG lalu base64 string."""
        with Image.open(path) as img:
            img = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- init ----------
    def __init__(
        self,
        images_b64: List[str],
        *,
        prompt: str = "",
        negative_prompt: str = "",
        steps: int = 30,
        cfg_scale: float = 7,
        denoising_strength: float = 0.75,
        sampler_name: str = "DPM++ 2M Karras",
        output_dir: str = "result",
        **kw
    ):
        if not images_b64:
            raise ValueError("images_b64 tidak boleh kosong")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.payload = {
            "init_images": images_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_name": sampler_name,
            "sampler_index": sampler_name,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "send_images": True,
            "save_images": False,
            **kw
        }

    # ---------- call ----------
    def generate(self, timeout: int = 300) -> Dict[str, Any]:
        r = requests.post(
            self.ENDPOINT,
            json=self.payload,
            timeout=timeout,
        )
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
        return r.json()

    def generate_and_save(self, timeout: int = 300) -> List[Dict[str, Any]]:
        start_time = time.time()
        resp = self.generate(timeout=timeout)
        elapsed_time = time.time() - start_time

        images = resp.get("images", [])
        metadata = []

        for idx, im_b64 in enumerate(images):
            filename = f"img2img_{idx}.png"
            path_file = os.path.join(self.output_dir, filename)

            with open(path_file, "wb") as f:
                f.write(base64.b64decode(im_b64))

            metadata.append({
                "img_base64": im_b64,
                "path_file": path_file,
                "elapsed_time": elapsed_time
            })

        # Simpan metadata ke JSON
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata


# -------------------------------------------------
# Contoh CLI
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools_generate_i2i.py </path/to/input.png>")
        sys.exit(1)

    input_img_path = sys.argv[1]
    img_b64 = SDImg2Img.file_to_base64(input_img_path)

    sd = SDImg2Img(
        images_b64=[img_b64],
        prompt="1girl, looking at viewer, masterpiece",
        negative_prompt="blurry, lowres, bad anatomy",
        steps=30,
        cfg_scale=7,
        denoising_strength=0.65,
    )

    metadata = sd.generate_and_save()
    print("Selesai, cek result/img2img_*.png dan result/metadata.json")