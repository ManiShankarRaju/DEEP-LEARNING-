# 🌟 Ultimate Photo Restoration

A powerful desktop application for restoring old and damaged photos using advanced computer vision techniques. Features intelligent crack removal, face enhancement, and professional-grade image restoration.

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🎯 Core Capabilities
- **Ultimate Face Clarity** - Advanced face detection and enhancement with eye detail optimization
- **Superior Crack Removal** - Multi-scale intelligent inpainting that preserves important features
- **Feature Preservation** - Smart algorithms that protect facial features during restoration
- **Professional Quality** - Multi-stage enhancement pipeline for maximum quality

### 🛠️ Restoration Methods

#### 👤 Face-Focused Restoration
- Ultimate Face Clarity - Maximum face enhancement with detail preservation
- Smart Face Enhancement - Balanced face improvement
- Eye & Detail Enhancement - Specialized eye and facial feature enhancement

#### 🔧 Crack & Damage Removal
- Intelligent Inpainting - Advanced crack detection and removal
- Multi-Scale Crack Fix - Processes cracks at multiple resolutions
- Feature-Preserving Fix - Removes damage while protecting important details

#### 🏆 Complete Restoration
- **ULTIMATE RESTORATION** - Full pipeline combining all techniques (5-8 minutes)
- Portrait Perfection - Optimized for portrait photos
- Professional Grade - Studio-quality restoration

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ultimate-photo-restoration.git
cd ultimate-photo-restoration
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### 🖥️ GUI Application (Recommended)

**Windows:**
```bash
ULTIMATE_RESTORATION.bat
```

**Mac/Linux:**
```bash
python ultimate_photo_restoration.py
```

#### 💻 Command Line Interface

**Basic usage:**
```bash
python ultimate_cli.py your_photo.jpg
```

**With specific mode:**
```bash
# Face enhancement only (2-3 minutes)
python ultimate_cli.py portrait.jpg face

# Crack removal only (3-4 minutes)
python ultimate_cli.py damaged.jpg crack

# Complete restoration (5-8 minutes)
python ultimate_cli.py old_photo.jpg ultimate
```

**Try with sample images:**
```bash
python ultimate_cli.py test_images/old_w_scratch/a.png ultimate
```

## 📸 How It Works

### The Ultimate Restoration Pipeline

1. **Multi-Scale Crack Detection**
   - Analyzes image at multiple resolutions
   - Uses morphological operations, edge detection, and adaptive thresholding
   - Combines multiple detection methods for accuracy

2. **Intelligent Inpainting**
   - Applies multiple inpainting algorithms (Telea, Navier-Stokes)
   - Intelligently blends results based on local image characteristics
   - Protects facial features from aggressive inpainting

3. **Face Detection & Enhancement**
   - Detects faces using Haar Cascades
   - Applies advanced CLAHE histogram equalization
   - Multi-scale processing for optimal detail
   - Specialized eye enhancement

4. **Final Quality Optimization**
   - Advanced noise reduction
   - Selective sharpening based on edge detection
   - Color enhancement in LAB color space
   - Bilateral filtering for smooth results

## 🎨 GUI Features

- **Visual Before/After Comparison** - See results side-by-side
- **Multiple Processing Options** - Choose from 9 different restoration methods
- **Customizable Settings**
  - Enhancement level (Conservative, Balanced, Maximum)
  - Focus areas (Faces, Eyes, Texture)
  - Processing options (Noise reduction, Color enhancement)
- **Sample Images** - Built-in test images to try
- **High-Quality Export** - Save results at maximum quality

## 📋 Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- scikit-image
- scipy
- tkinter (usually included with Python)

## 🖼️ Sample Results

The application works best with:
- Old family photos with cracks and scratches
- Faded portraits
- Damaged historical photographs
- Low-quality scanned images

## ⚙️ Advanced Options

### Enhancement Levels
- **Conservative** - Subtle improvements, minimal changes
- **Balanced** - Good balance between enhancement and naturalness
- **Maximum Quality** - Aggressive enhancement for best results

### Processing Time
- Face enhancement: 2-3 minutes
- Crack removal: 3-4 minutes
- Ultimate restoration: 5-8 minutes

*Times vary based on image size and computer performance*

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project uses computer vision techniques inspired by research in photo restoration, including:
- OpenCV library for image processing
- Haar Cascades for face detection
- Advanced inpainting algorithms

## 💡 Tips for Best Results

1. **For portraits**: Use "Ultimate Face Clarity" or "Portrait Perfection"
2. **For damaged photos**: Use "ULTIMATE RESTORATION" for complete fix
3. **For quick fixes**: Use "Smart Face Enhancement" or "Intelligent Inpainting"
4. **Image size**: Works best with images between 500x500 and 2000x2000 pixels
5. **Processing time**: Be patient with "ULTIMATE RESTORATION" - quality takes time!

## 🐛 Troubleshooting

**GUI won't start:**
- Make sure tkinter is installed: `python -m tkinter`
- Try running from command line to see error messages

**Out of memory errors:**
- Resize large images before processing
- Close other applications
- Use CLI mode instead of GUI

**Poor results:**
- Try different enhancement levels
- Experiment with different restoration methods
- Some images may need manual touch-up after restoration

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for preserving precious memories**
