import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import argparse
import os


def load_image(image_path):
    """이미지를 로드하고 numpy array로 변환"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    # RGB로 변환 (RGBA나 다른 모드일 경우)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def calculate_psnr(img1_path, img2_path):
    """
    두 이미지 간의 PSNR을 계산
    
    Args:
        img1_path: 첫 번째 이미지 경로
        img2_path: 두 번째 이미지 경로
    
    Returns:
        PSNR 값 (dB)
    """
    # 이미지 로드
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # 이미지 크기가 다른 경우 리사이즈
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes are different. Resizing...")
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # 두 번째 이미지를 첫 번째 이미지 크기로 리사이즈
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
        img2 = np.array(img2_pil)
    
    # 이미지를 float 타입으로 변환하고 0-1 범위로 정규화
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    
    # PSNR 계산
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    
    return psnr_value


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR between two images')
    parser.add_argument('--img1', type=str, 
                       default='./img/d[0048]TopBF1.png',
                       help='Path to first image')
    parser.add_argument('--img2', type=str,
                       default='./result/result_2024-05-12T21-21-24_new.png',
                       help='Path to second image')
    
    args = parser.parse_args()
    
    try:
        psnr = calculate_psnr(args.img1, args.img2)
        print(f"\n{'='*50}")
        print(f"Image 1: {args.img1}")
        print(f"Image 2: {args.img2}")
        print(f"{'='*50}")
        print(f"PSNR: {psnr:.4f} dB")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

