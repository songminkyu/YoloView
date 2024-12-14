import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Tuple, Dict, List
from skimage.metrics import structural_similarity as ssim
from utils.brisque.brisque import BRISQUE
import scipy

class ImageQualityEvaluator:
    def __init__(self, blur_threshold=100.0, noise_threshold=20.0, jpeg_artifact_threshold=0.5, model_path=None):
        """
        이미지 품질 평가를 위한 초기화.
        Args:
            blur_threshold (float): 블러 감지 임계값
            noise_threshold (float): 노이즈 감지 임계값
            jpeg_artifact_threshold (float): JPEG 압축 아티팩트 감지 임계값
            model_path (str): 딥러닝 모델 경로 (품질 예측)
        """
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold
        self.jpeg_artifact_threshold = jpeg_artifact_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path:
            self.quality_model = self.load_model(model_path)
        else:
            self.quality_model = None

    def load_model(self, model_path: str) -> nn.Module:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)  # 품질 점수 예측 (단일 출력)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def detect_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.blur_threshold
        return is_blurry, laplacian_var

    def detect_noise(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_signal = np.mean(gray)
        noise = np.std(gray)
        snr = mean_signal / (noise + 1e-10)
        is_noisy = snr < self.noise_threshold
        return is_noisy, snr

    def detect_jpeg_artifacts(self, image: np.ndarray) -> Tuple[bool, float]:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 50]
        _, compressed = cv2.imencode('.jpg', image, encode_param)
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

        gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_compressed = cv2.cvtColor(decompressed, cv2.COLOR_BGR2GRAY)
        artifact_score = ssim(gray_original, gray_compressed)
        is_artifacted = artifact_score < self.jpeg_artifact_threshold
        return is_artifacted, artifact_score

    def calculate_psnr(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mse = np.mean((gray - gray.mean()) ** 2)
        if mse == 0:
            return float('inf')  # 최대 PSNR
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr

    def calculate_entropy(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_prob = hist / hist.sum()
        entropy = -np.sum([p * np.log2(p) for p in hist_prob if p > 0])
        return entropy

    def calculate_brisque(self, image: np.ndarray) -> float:
        """
        BRISQUE 품질 점수 계산 (낮을수록 품질이 좋음).
        Args:
            image (np.ndarray): 입력 이미지
        Returns:
            float: BRISQUE 점수
        """
        brisque_score = BRISQUE(url=False)
        score = brisque_score.score(image)
        return score

    def evaluate(self, image_path: str) -> Dict[str, Tuple[bool, float]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")

        results = {}
        results["blur"] = self.detect_blur(image)
        results["noise"] = self.detect_noise(image)
        results["jpeg_artifacts"] = self.detect_jpeg_artifacts(image)
        results["psnr"] = self.calculate_psnr(image)
        results["entropy"] = self.calculate_entropy(image)
        results["brisque"] = (None, self.calculate_brisque(image))

        if self.quality_model:
            results["dl_quality"] = (None, self.predict_quality_with_model(image))

        return results

# 유틸리티 함수
def print_evaluation_results(results: Dict[str, Tuple[bool, float]], image_name: str = "Image"):
    print(f"{image_name} Quality Evaluation Results:")
    for key, (status, metric) in results.items():
        if key in ["brisque", "dl_quality"]:
            print(f"  {key.capitalize()}: {metric:.2f}")
        else:
            print(f"  {key.capitalize()}: {status} (Metric: {metric:.2f})")
    print()

# 실행 예시
if __name__ == "__main__":
    evaluator = ImageQualityEvaluator(
        blur_threshold=100.0,
        noise_threshold=20.0,
        jpeg_artifact_threshold=0.6
    )

    image_path = "c:\\Users\\USER\\Downloads\\rock-paper-scissors.v14i.yolov11\\train\\images\\IMG_7077_MOV-152_jpg.rf.471fd18f355fc477a4b568473d1da01b.jpg"
    try:
        result = evaluator.evaluate(image_path)
        print_evaluation_results(result, image_name="Single Image")
    except FileNotFoundError as e:
        print(e)
