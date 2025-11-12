# 📸 AI Photo Profile Generator

AI-powered social media profile photo generator dengan kemampuan text-to-image dan image-to-image transformation menggunakan Stable Diffusion dan OpenAI.

## ✨ Fitur

### 🎨 Text-to-Image Generator
- Generate foto profil dari deskripsi teks
- Multiple style options (professional, casual, creative, artistic)
- Various aspect ratios (1:1, 4:5, 16:9)
- High-resolution output

### 🔄 Image-to-Image Transformer
- Enhance existing photos
- Style transfer untuk foto profil
- Background replacement
- Quality improvement


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU recommended (dapat run di CPU tapi lebih lambat)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/hannesegi/agent-photo-generator.git
cd agent-photo-generator.git
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables**
```bash
cp .ini
```
Edit `.ini` file:
```ini


```

4. **Run the application service**
```bash
python main_service_img2img.py
python main_service_photo_gent2i.py

```

Akses aplikasi di: `http://localhost:7028`


## 🎨 Contoh Prompts

### Professional Headshots
```
Professional corporate headshot, business attire, clean background, professional lighting, sharp focus, 8k resolution
```

### Creative Profiles
```
Artistic portrait, creative background, soft lighting, cinematic look, professional photographer, high detail
```

### Casual Social Media
```
Casual outdoor portrait, natural lighting, smiling, authentic expression, social media style, high quality
```

## 🛠️ API Endpoints

### Generate Image from Text
```http
POST /generate-photo-profile
Content-Type: application/json

{
  "prompt": "professional headshot of software developer",
}
```
