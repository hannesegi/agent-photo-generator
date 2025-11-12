#!/usr/bin/env python3
import base64
import json
import pathlib
from typing import Dict, Any
import srsly
import requests

class SDClientT2I:
    """
    Client minimal AUTOMATIC1111 WebUI txt2img.
    Checkpoint hard-coded ke realisticUniversalBase_100.safetensors
    """

    def __init__(self, base_url: str = ""):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/sdapi/v1/txt2img"
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "Content-Type": "application/json"})

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """Payload default + prompt + checkpoint hard-coded"""
        return {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted face, extra limbs, watermark, text, logo, disabled, deformed, disfigured, bad anatomy, more than one person, multiple people",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": "DPM++ 2M Karras",
            "scheduler": "Karras",
            "batch_size": 1,
            "n_iter": 1,
            "steps": 30,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "restore_faces": True,
            "tiling": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "eta": 0,
            "denoising_strength": 0,
            "override_settings": {
                "sd_model_checkpoint": "realisticUniversalBase_100.safetensors"
            },
            "override_settings_restore_afterwards": True,
            "refiner_checkpoint": "",
            "refiner_switch_at": 0.8,
            "disable_extra_networks": False,
            "firstpass_image": "",
            "comments": {},
            "enable_hr": False,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": "Latent",
            "hr_second_pass_steps": 20,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "hr_checkpoint_name": "",
            "hr_sampler_name": "DPM++ 2M Karras",
            "hr_scheduler": "Karras",
            "hr_prompt": "",
            "hr_negative_prompt": "",
            "force_task_id": "",
            "sampler_index": "DPM++ 2M Karras",
            "script_name": "",
            "script_args": [],
            "send_images": True,
            "save_images": True,
            "alwayson_scripts": {},
            "infotext": ""
        }

    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate satu gambar dari prompt string.
        Return dict: {"base64": <str>, "path": <str>}
        """
        payload = self._build_payload(prompt)
        response = self.session.post(self.endpoint, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()

        # simpan file
        out_file = pathlib.Path("output") / f"{hash(prompt) % 1000000}.png"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        png_bytes = base64.b64decode(data["images"][0])
        out_file.write_bytes(png_bytes)

        return {
            "base64": data["images"][0],
            "path": str(out_file.resolve())
        }


if __name__ == "__main__":
    client = SDClientT2I(base_url="")
    result = client.generate(
        "create a picture profile one young man, realistic, clean background, soft lighting, high detail"
    )
    import srsly
    srsly.write_json("outputs_test.json",result)