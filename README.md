# 🌟 Ultimate Photo Restoration

A powerful desktop application for restoring old and damaged photos using advanced computer vision techniques. Features intelligent crack removal, face enhancement, and professional-grade image restoration.

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **Inspired by:** [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) (CVPR 2020)  
> This project implements a lightweight, user-friendly version using classical computer vision techniques instead of deep learning.

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
git clone https://github.com/YOUR_GITHUB_USERNAME/ultimate-photo-restoration.git
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

This implementation uses **classical computer vision techniques** (no deep learning required) for accessible photo restoration:

1. **Multi-Scale Crack Detection**
   - Analyzes image at multiple resolutions (1.0x, 0.8x, 0.6x, 0.4x)
   - Uses morphological operations (tophat, blackhat)
   - Edge detection (Canny, Sobel, Laplacian)
   - Adaptive thresholding for crack identification
   - Combines multiple detection methods with weighted blending

2. **Intelligent Inpainting**
   - Applies multiple inpainting algorithms:
     - **Telea (Fast Marching Method)** - Better for textured areas
     - **Navier-Stokes** - Better for smooth areas
     - **Large radius Telea** - Better for structure preservation
   - Intelligently blends results based on local variance and edge strength
   - Protects facial features from aggressive inpainting

3. **Face Detection & Enhancement**
   - Detects faces using OpenCV Haar Cascades
   - Applies advanced CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Multi-scale processing (0.5x, 1.0x, 1.5x, 2.0x) for optimal detail
   - Specialized eye enhancement with targeted sharpening
   - Unsharp masking at multiple frequencies
   - Bilateral filtering for skin smoothing

4. **Final Quality Optimization**
   - Advanced noise reduction (Non-local Means Denoising)
   - Selective sharpening based on edge detection
   - Color enhancement in LAB color space
   - Bilateral filtering for smooth results
   - Adaptive processing based on local image characteristics

**Advantages of This Approach:**
- ✅ No GPU required
- ✅ No large model downloads
- ✅ Fast processing (2-8 minutes)
- ✅ Works offline
- ✅ Easy to understand and modify

**Trade-offs vs Deep Learning:**
- ⚠️ Less sophisticated than state-of-the-art deep learning models
- ⚠️ May not handle extreme degradation as well
- ✅ But much more accessible and practical for everyday use!

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

**Note:** This is an independent implementation using classical computer vision techniques. The original "Bringing Old Photos Back to Life" project by Microsoft Research is also under MIT License.

## 🙏 Acknowledgments

This project is inspired by the groundbreaking research:

**"Bringing Old Photos Back to Life"** (CVPR 2020, Oral)  
**"Old Photo Restoration via Deep Latent Space Translation"** (TPAMI 2022)

*Authors:* Ziyu Wan, Bo Zhang, Dongdong Chen, Pan Zhang, Dong Chen, Jing Liao, Fang Wen  
*Affiliations:* City University of Hong Kong, Microsoft Research Asia, Microsoft Cloud AI, USTC

**Citation:**
```bibtex
@inproceedings{wan2020bringing,
  title={Bringing Old Photos Back to Life},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2747--2757},
  year={2020}
}

@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

**Key Differences:**
- **Original Project:** Uses deep learning (PyTorch) with pretrained models for state-of-the-art results
- **This Project:** Uses classical computer vision (OpenCV) for lightweight, accessible restoration without GPU requirements

**Technologies Used:**
- OpenCV library for image processing
- Haar Cascades for face detection
- Advanced inpainting algorithms (Telea, Navier-Stokes)
- Multi-scale image processing techniques

**Related Resources:**
- [Original Project](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
- [Project Page](http://raywzy.com/Old_Photo/)
- [Colab Demo](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing)
- [Old Film Restoration](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) (CVPR 2022)

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
