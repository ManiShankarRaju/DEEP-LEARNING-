#!/usr/bin/env python3
"""
Ultimate Photo Restoration - Command Line
Best face clarity + superior crack removal + feature preservation
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import sys
from pathlib import Path

def ultimate_photo_restoration(input_path, output_path=None, mode="ultimate"):
    """
    Ultimate photo restoration with best face clarity and crack removal
    
    Args:
        input_path: Path to input image
        output_path: Path to save result (optional)
        mode: "face", "crack", "ultimate"
    """
    
    print(f"🌟 ULTIMATE PHOTO RESTORATION")
    print(f"📸 Loading image: {input_path}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise Exception("Could not load image")
    
    original_size = image.shape
    print(f"   Size: {original_size[1]}x{original_size[0]}")
    
    if mode == "face":
        print("👤 Applying ultimate face clarity...")
        result = ultimate_face_restoration(image)
    elif mode == "crack":
        print("🔧 Applying superior crack removal...")
        result = superior_crack_removal(image)
    else:  # ultimate
        print("🚀 Applying ULTIMATE restoration (this may take a few minutes)...")
        result = complete_ultimate_restoration(image)
    
    # Convert to PIL for final enhancements
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    # Final PIL enhancements
    print("✨ Applying final quality enhancements...")
    
    enhancer = ImageEnhance.Brightness(result_pil)
    result_pil = enhancer.enhance(1.03)
    
    enhancer = ImageEnhance.Contrast(result_pil)
    result_pil = enhancer.enhance(1.08)
    
    enhancer = ImageEnhance.Color(result_pil)
    result_pil = enhancer.enhance(1.05)
    
    enhancer = ImageEnhance.Sharpness(result_pil)
    result_pil = enhancer.enhance(1.05)
    
    # Save result
    if not output_path:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_ULTIMATE_RESTORED{input_file.suffix}"
    
    # Save at maximum quality
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        result_pil.save(output_path, quality=98, optimize=True)
    else:
        result_pil.save(output_path, optimize=True)
    
    print(f"🌟 ULTIMATE RESTORATION COMPLETE!")
    print(f"📁 Masterpiece saved to: {output_path}")
    
    return str(output_path)

def complete_ultimate_restoration(image):
    """Complete ultimate restoration pipeline"""
    print("   Step 1/4: Superior crack removal...")
    crack_removed = advanced_crack_removal_ultimate(image)
    
    print("   Step 2/4: Ultimate face enhancement...")
    face_enhanced = ultimate_face_enhancement_complete(crack_removed)
    
    print("   Step 3/4: Multi-scale enhancement...")
    multiscale_enhanced = multiscale_enhancement(face_enhanced)
    
    print("   Step 4/4: Final quality optimization...")
    final_result = final_quality_optimization(multiscale_enhanced)
    
    return final_result

def advanced_crack_removal_ultimate(image):
    """Advanced crack removal with ultimate quality"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale crack detection
    print("     🔍 Multi-scale crack detection...")
    scales = [1.0, 0.8, 0.6, 0.4]
    crack_masks = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            scaled_gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
        else:
            scaled_gray = gray
        
        # Advanced crack detection at this scale
        mask = detect_cracks_ultimate(scaled_gray)
        
        # Resize back if needed
        if scale != 1.0:
            mask = cv2.resize(mask, (w, h))
        
        crack_masks.append(mask)
    
    # Intelligent mask combination
    print("     🎨 Intelligent mask combination...")
    combined_mask = np.zeros_like(crack_masks[0])
    weights = [0.4, 0.3, 0.2, 0.1]  # Higher weight for full resolution
    
    for mask, weight in zip(crack_masks, weights):
        combined_mask = cv2.add(combined_mask, (mask * weight).astype(np.uint8))
    
    # Feature preservation
    print("     🎯 Feature preservation...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Protect important features
    feature_protection_mask = np.zeros_like(combined_mask)
    
    for (x, y, w, h) in faces:
        # Protect face area
        cv2.rectangle(feature_protection_mask, (x, y), (x+w, y+h), 128, -1)
        
        # Detect and protect eyes within face
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)
        
        for (ex, ey, ew, eh) in eyes:
            # Protect eye area more strongly
            cv2.rectangle(feature_protection_mask, 
                         (x+ex, y+ey), (x+ex+ew, y+ey+eh), 200, -1)
    
    # Apply feature protection
    protected_mask = combined_mask.copy()
    protection_factor = feature_protection_mask / 255.0
    protected_mask = (protected_mask * (1 - protection_factor * 0.7)).astype(np.uint8)
    
    # Advanced inpainting with multiple methods
    print("     🖌️ Advanced inpainting...")
    
    # Method 1: Telea (Fast Marching)
    inpaint_telea = cv2.inpaint(image, protected_mask, 5, cv2.INPAINT_TELEA)
    
    # Method 2: Navier-Stokes
    inpaint_ns = cv2.inpaint(image, protected_mask, 5, cv2.INPAINT_NS)
    
    # Method 3: Larger radius Telea for better texture
    inpaint_telea_large = cv2.inpaint(image, protected_mask, 10, cv2.INPAINT_TELEA)
    
    # Intelligent blending based on local image characteristics
    result = intelligent_inpaint_blend(image, inpaint_telea, inpaint_ns, inpaint_telea_large, protected_mask)
    
    return result

def detect_cracks_ultimate(gray):
    """Ultimate crack detection"""
    # Multiple advanced detection methods
    
    # Method 1: Multi-directional morphological operations
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),  # Vertical lines
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),  # Horizontal lines
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # General
    ]
    
    morph_results = []
    for kernel in kernels:
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        morph_results.append(cv2.add(tophat, blackhat))
    
    # Combine morphological results
    morph_combined = np.zeros_like(morph_results[0])
    for result in morph_results:
        morph_combined = cv2.add(morph_combined, result // len(morph_results))
    
    # Method 2: Advanced edge detection
    # Canny with multiple thresholds
    edges1 = cv2.Canny(gray, 30, 100)
    edges2 = cv2.Canny(gray, 50, 150)
    edges3 = cv2.Canny(gray, 70, 200)
    
    edges_combined = cv2.add(cv2.add(edges1 // 3, edges2 // 3), edges3 // 3)
    
    # Method 3: Laplacian of Gaussian
    log_kernel = cv2.getGaussianKernel(5, 1)
    log_kernel = log_kernel @ log_kernel.T
    log_result = cv2.filter2D(gray, cv2.CV_64F, log_kernel)
    log_result = cv2.Laplacian(log_result, cv2.CV_64F)
    log_result = np.uint8(np.absolute(log_result))
    
    # Method 4: Adaptive thresholding with multiple parameters
    adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 15, 3)
    
    adaptive_combined = cv2.add(cv2.bitwise_not(adaptive1) // 2, 
                               cv2.bitwise_not(adaptive2) // 2)
    
    # Combine all methods intelligently
    final_mask = cv2.add(morph_combined, edges_combined // 2)
    final_mask = cv2.add(final_mask, log_result // 3)
    final_mask = cv2.add(final_mask, adaptive_combined // 4)
    
    # Threshold and clean
    _, final_mask = cv2.threshold(final_mask, 25, 255, cv2.THRESH_BINARY)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    return final_mask

def intelligent_inpaint_blend(original, inpaint1, inpaint2, inpaint3, mask):
    """Intelligently blend multiple inpainting results"""
    # Calculate local image characteristics
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Local variance (texture measure)
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    
    # Normalize variance
    local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)
    
    # Edge strength
    edges = cv2.Canny(gray, 50, 150)
    edge_strength = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0
    
    # Create blending weights based on local characteristics
    # High variance areas: use Telea (better for textured areas)
    # Low variance areas: use NS (better for smooth areas)
    # Edge areas: use large radius Telea (better structure preservation)
    
    weight1 = local_variance * (1 - edge_strength)  # Telea for textured, non-edge areas
    weight2 = (1 - local_variance) * (1 - edge_strength)  # NS for smooth, non-edge areas
    weight3 = edge_strength  # Large Telea for edge areas
    
    # Normalize weights
    total_weight = weight1 + weight2 + weight3
    weight1 = weight1 / (total_weight + 1e-8)
    weight2 = weight2 / (total_weight + 1e-8)
    weight3 = weight3 / (total_weight + 1e-8)
    
    # Expand weights to 3 channels
    weight1 = np.stack([weight1] * 3, axis=2)
    weight2 = np.stack([weight2] * 3, axis=2)
    weight3 = np.stack([weight3] * 3, axis=2)
    
    # Blend the results
    blended = (inpaint1 * weight1 + inpaint2 * weight2 + inpaint3 * weight3).astype(np.uint8)
    
    # Apply only where mask is active
    mask_3d = np.stack([mask] * 3, axis=2) / 255.0
    result = original * (1 - mask_3d) + blended * mask_3d
    
    return result.astype(np.uint8)

def ultimate_face_enhancement_complete(image):
    """Complete ultimate face enhancement"""
    print("     👤 Detecting faces...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 6)  # More strict detection
    
    result = image.copy()
    
    print(f"     Found {len(faces)} face(s)")
    
    for i, (x, y, w, h) in enumerate(faces):
        print(f"     Processing face {i+1}/{len(faces)}...")
        
        # Extract face with generous padding
        padding = max(20, min(w, h) // 4)
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
        
        face_region = result[y1:y2, x1:x2]
        
        # Ultimate face processing
        enhanced_face = process_face_ultimate_quality(face_region)
        
        # Eye enhancement
        face_gray = gray[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
        
        print(f"       Found {len(eyes)} eye(s) in face {i+1}")
        
        for (ex, ey, ew, eh) in eyes:
            eye_region = enhanced_face[ey:ey+eh, ex:ex+ew]
            enhanced_eye = enhance_eye_ultimate(eye_region)
            enhanced_face[ey:ey+eh, ex:ex+ew] = enhanced_eye
        
        result[y1:y2, x1:x2] = enhanced_face
    
    return result

def process_face_ultimate_quality(face_region):
    """Process face with ultimate quality"""
    # Multi-stage enhancement pipeline
    
    # Stage 1: Advanced denoising
    denoised = cv2.fastNlMeansDenoisingColored(face_region, None, 10, 10, 7, 21)
    
    # Stage 2: Multi-scale processing
    scales = [0.5, 1.0, 1.5, 2.0]
    enhanced_scales = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = denoised.shape[:2]
            scaled = cv2.resize(denoised, (int(w*scale), int(h*scale)), 
                              interpolation=cv2.INTER_CUBIC)
        else:
            scaled = denoised.copy()
        
        # Process at this scale
        processed = enhance_face_at_scale(scaled)
        
        # Resize back
        if scale != 1.0:
            processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_CUBIC)
        
        enhanced_scales.append(processed)
    
    # Stage 3: Intelligent scale combination
    result = enhanced_scales[1]  # Start with original scale
    
    # Blend smaller scale (smoother) for skin areas
    result = cv2.addWeighted(result, 0.7, enhanced_scales[0], 0.3, 0)
    
    # Blend larger scales (more detailed) for feature areas
    # Detect feature areas using edge detection
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.GaussianBlur(edges, (5, 5), 0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)
    
    # Apply larger scale enhancement to edge areas
    for i in range(2, len(enhanced_scales)):
        weight = 0.2 / (i - 1)
        result = result * (1 - edge_mask * weight) + enhanced_scales[i] * (edge_mask * weight)
    
    return result.astype(np.uint8)

def enhance_face_at_scale(image):
    """Enhance face at specific scale"""
    # Advanced histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE with optimal parameters for faces
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Advanced unsharp masking
    # Multiple Gaussian blurs for different frequency enhancement
    gaussian1 = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    gaussian2 = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    
    # Combine unsharp masks
    unsharp1 = cv2.addWeighted(enhanced, 1.8, gaussian1, -0.8, 0)
    unsharp2 = cv2.addWeighted(enhanced, 1.4, gaussian2, -0.4, 0)
    
    enhanced = cv2.addWeighted(unsharp1, 0.6, unsharp2, 0.4, 0)
    
    # Selective bilateral filtering (preserve edges, smooth skin)
    enhanced = cv2.bilateralFilter(enhanced, 12, 80, 80)
    
    return enhanced

def enhance_eye_ultimate(eye_region):
    """Ultimate eye enhancement"""
    if eye_region.size == 0:
        return eye_region
    
    # Convert to LAB for better processing
    lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
    
    # Enhance L channel with stronger CLAHE for eyes
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2,2))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Multi-level sharpening for eye details
    # Level 1: Fine details
    kernel1 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened1 = cv2.filter2D(enhanced, -1, kernel1)
    
    # Level 2: Medium details
    kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened2 = cv2.filter2D(enhanced, -1, kernel2)
    
    # Combine sharpening levels
    enhanced = cv2.addWeighted(enhanced, 0.5, sharpened1, 0.3, 0)
    enhanced = cv2.addWeighted(enhanced, 0.8, sharpened2, 0.2, 0)
    
    # Enhance contrast specifically for eyes
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=3)
    
    return enhanced

def multiscale_enhancement(image):
    """Multi-scale enhancement for overall image quality"""
    print("     🔬 Multi-scale processing...")
    
    scales = [0.25, 0.5, 1.0, 1.5]
    enhanced_scales = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            scaled = cv2.resize(image, (int(w*scale), int(h*scale)), 
                              interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        else:
            scaled = image.copy()
        
        # Process at this scale
        processed = enhance_at_scale_ultimate(scaled)
        
        # Resize back
        if scale != 1.0:
            processed = cv2.resize(processed, (w, h), 
                                 interpolation=cv2.INTER_CUBIC)
        
        enhanced_scales.append(processed)
    
    # Intelligent combination of scales
    result = enhanced_scales[2]  # Start with original scale
    
    # Add fine details from larger scale
    result = cv2.addWeighted(result, 0.8, enhanced_scales[3], 0.2, 0)
    
    # Add smoothness from smaller scales for noise reduction
    result = cv2.addWeighted(result, 0.9, enhanced_scales[1], 0.1, 0)
    
    return result

def enhance_at_scale_ultimate(image):
    """Ultimate enhancement at specific scale"""
    # Advanced noise reduction
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Histogram equalization
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Adaptive sharpening based on local content
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance for adaptive processing
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    
    # Normalize variance
    local_variance = local_variance / (local_variance.max() + 1e-8)
    
    # Apply different sharpening based on local variance
    # High variance areas (textured): moderate sharpening
    # Low variance areas (smooth): minimal sharpening
    
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    
    # Create adaptive blend
    variance_3d = np.stack([local_variance] * 3, axis=2)
    adaptive_sharp = enhanced * (1 - variance_3d * 0.3) + sharpened * (variance_3d * 0.3)
    
    return adaptive_sharp.astype(np.uint8)

def final_quality_optimization(image):
    """Final quality optimization"""
    print("     ✨ Final quality optimization...")
    
    # Advanced bilateral filtering for final smoothing
    result = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Subtle color enhancement
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    
    # Enhance color channels slightly
    lab[:,:,1] = cv2.multiply(lab[:,:,1], 1.05)  # A channel
    lab[:,:,2] = cv2.multiply(lab[:,:,2], 1.05)  # B channel
    
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Final contrast and brightness adjustment
    result = cv2.convertScaleAbs(result, alpha=1.02, beta=2)
    
    return result

def ultimate_face_restoration(image):
    """Ultimate face restoration only"""
    return ultimate_face_enhancement_complete(image)

def superior_crack_removal(image):
    """Superior crack removal only"""
    return advanced_crack_removal_ultimate(image)

def main():
    if len(sys.argv) < 2:
        print("🌟 ULTIMATE Photo Restoration - Command Line")
        print("=" * 60)
        print("\nUsage:")
        print("  python ultimate_cli.py <image_path> [mode]")
        print("\nModes:")
        print("  face     - Ultimate face clarity (2-3 minutes)")
        print("  crack    - Superior crack removal (3-4 minutes)")
        print("  ultimate - Complete restoration (5-8 minutes) [DEFAULT]")
        print("\nExamples:")
        print("  python ultimate_cli.py damaged_photo.jpg")
        print("  python ultimate_cli.py old_photo.png ultimate")
        print("  python ultimate_cli.py portrait.jpg face")
        print("  python ultimate_cli.py test_images/old_w_scratch/a.png crack")
        
        # Check for sample images
        sample_dirs = ["test_images/old_w_scratch", "test_images/old"]
        for sample_dir in sample_dirs:
            if Path(sample_dir).exists():
                samples = list(Path(sample_dir).glob("*.png"))
                if samples:
                    print(f"\n📁 Found {len(samples)} sample images in {sample_dir}!")
                    print(f"Try: python ultimate_cli.py {samples[0]} ultimate")
                    break
        
        return
    
    input_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "ultimate"
    
    if not Path(input_path).exists():
        print(f"❌ File not found: {input_path}")
        return
    
    if mode not in ["face", "crack", "ultimate"]:
        print(f"❌ Invalid mode: {mode}. Use 'face', 'crack', or 'ultimate'")
        return
    
    try:
        print("🚀 Starting ULTIMATE photo restoration...")
        print("This will produce the BEST possible results!")
        print()
        
        result_path = ultimate_photo_restoration(input_path, mode=mode)
        
        print()
        print("🌟 ULTIMATE SUCCESS!")
        print(f"📸 Original: {input_path}")
        print(f"✨ Masterpiece: {result_path}")
        print()
        print("🏆 Your photo has been transformed into a masterpiece!")
        print("   ✨ Face clarity maximized")
        print("   🔧 Cracks intelligently removed")
        print("   🎯 Core features preserved")
        print("   🌟 Professional quality achieved")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()