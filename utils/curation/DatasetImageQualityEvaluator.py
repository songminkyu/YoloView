import cv2
import numpy as np
import torch
import torch.nn as nn
import pyiqa
from torchvision import transforms, models
from typing import Tuple, Dict
from utils.brisque.brisque import BRISQUE
import os


class ImageLoader:
    """이미지를 로드하고 유효성을 검사하는 클래스."""

    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image at path {image_path} not found.")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}.")
        return image


class BlurDetector:
    """이미지 블러 여부 감지 클래스."""

    def __init__(self, image_path : str ,threshold: float = 100.0):
        """
        Args:
            threshold (float): 블러 감지 임계값. 이 값보다 작으면 블러로 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def detect_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.threshold
        return is_blurry, laplacian_var

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.detect_blur(image)


class NoiseDetector:
    """이미지 노이즈 감지 클래스."""

    def __init__(self, image_path : str ,threshold: float = 20.0):
        """
        Args:
            threshold (float): SNR 임계값. 이 값보다 SNR이 낮으면 노이즈가 많다고 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def detect_noise(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_signal = np.mean(gray)
        noise = np.std(gray)
        snr = mean_signal / (noise + 1e-10)
        is_noisy = snr < self.threshold
        return is_noisy, snr

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.detect_noise(image)


class EntropyCalculator:
    """이미지 엔트로피 계산 클래스."""

    def __init__(self,image_path : str, threshold: float = 4.0):
        """
        Args:
            threshold (float): 엔트로피 임계값. 엔트로피가 이 값보다 크면 분포가 다양하다고 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def calculate_entropy(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_prob = hist / hist.sum()
        entropy = -np.sum([p * np.log2(p) for p in hist_prob if p > 0])
        is_good = entropy > self.threshold
        return is_good, entropy

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.calculate_entropy(image)


class BRISQUECalculator:
    """BRISQUE 점수 계산 클래스."""

    def __init__(self, image_path : str, threshold: float = 50.0):
        """
        Args:
            threshold (float): BRISQUE 임계값. 점수가 이 값보다 작을수록 품질이 좋다고 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def calculate_brisque(self, image: np.ndarray) -> Tuple[bool, float]:
        try:
            brisque_score = BRISQUE(url=False)
            score = brisque_score.score(image)
        except Exception as e:
            # BRISQUE 계산 시 에러 처리
            raise RuntimeError(f"BRISQUE calculation failed: {e}")

        is_good = score < self.threshold
        return is_good, score

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.calculate_brisque(image)


class NIQECalculator:
    """NIQE 점수 계산 클래스."""

    def __init__(self,image_path : str, threshold: float = 5.0):
        """
        Args:
            threshold (float): NIQE 임계값. 점수가 이 값보다 작을수록 품질이 좋다고 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def calculate_niqe(self, image: np.ndarray) -> Tuple[bool, float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # GPU가 없는 경우 CPU로 fallback
        im_t = torch.FloatTensor(image.transpose(2, 0, 1)[None]) / 255.0
        im_t = im_t.to(device)

        try:
            niqe_metric = pyiqa.create_metric("niqe", device=device)
            score_tensor = niqe_metric(im_t)
            score = score_tensor.item()
        except Exception as e:
            raise RuntimeError(f"NIQE calculation failed: {e}")

        is_good = score < self.threshold
        return is_good, score

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.calculate_niqe(image)


class DLQualityPredictor:
    """딥러닝 모델을 활용한 이미지 품질 예측 클래스."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Args:
            model_path (str): 딥러닝 모델 파라미터 경로
            threshold (float): 예측 품질 점수에 대한 임계값. 이 값보다 크면 품질이 좋다고 판단.
        """
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str) -> nn.Module:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Error loading model state_dict: {e}")

        model.to(self.device)
        model.eval()
        return model

    def predict_quality_with_model(self, image: np.ndarray) -> Tuple[bool, float]:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            quality_score = self.model(image_tensor).item()

        is_good = quality_score > self.threshold
        return is_good, quality_score


class ImageQualityEvaluator:
    """이미지 품질 평가를 종합하는 클래스."""

    def __init__(self,
                 image_path : str,
                 blur_threshold=100.0,
                 noise_threshold=20.0,
                 entropy_threshold=4.0,
                 brisque_threshold=50.0,
                 niqe_threshold=5.0,
                 model_path=None):
        """
        Args:
            blur_threshold (float): 블러 감지 임계값
            noise_threshold (float): 노이즈 감지 임계값
            entropy_threshold (float): 엔트로피 임계값
            brisque_threshold (float): BRISQUE 임계값
            niqe_threshold (float): NIQE 임계값
            model_path (str): 딥러닝 모델 경로 (품질 예측용)
        """
        self.image_path = image_path
        self.blur_detector = BlurDetector(image_path="", threshold=blur_threshold)
        self.noise_detector = NoiseDetector(image_path="", threshold=noise_threshold)
        self.entropy_calc = EntropyCalculator(image_path="", threshold=entropy_threshold)
        self.brisque_calc = BRISQUECalculator(image_path="", threshold=brisque_threshold)
        self.niqe_calc = NIQECalculator(image_path="", threshold=niqe_threshold)

        if model_path:
            self.dl_predictor = DLQualityPredictor(model_path=model_path, threshold=0.5)
        else:
            self.dl_predictor = None

    def evaluate(self) -> Dict[str, Tuple[bool, float]]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        results = {}

        # Blur 평가
        results["blur"] = self.blur_detector.detect_blur(image)

        # Noise 평가
        results["noise"] = self.noise_detector.detect_noise(image)

        # NIQE 평가
        results["niqe"] = self.niqe_calc.calculate_niqe(image)

        # Entropy 평가
        results["entropy"] = self.entropy_calc.calculate_entropy(image)

        # BRISQUE 평가
        results["brisque"] = self.brisque_calc.calculate_brisque(image)

        # 딥러닝 모델 기반 품질 평가
        if self.dl_predictor:
            results["dl_quality"] = self.dl_predictor.predict_quality_with_model(image)

        return results

    def print_evaluation_results(self, results: Dict[str, Tuple[bool, float]], image_name: str = "Image"):
        """평가 결과를 콘솔에 출력하는 함수."""
        print(f"{image_name} Quality Evaluation Results:")
        for key, (status, metric) in results.items():
            print(f"  {key.capitalize()}: {status} (Score: {metric:.2f})")
        print()


# 실행 예시
if __name__ == "__main__":
    image_path = "c:\\Users\\USER\\Desktop\\M1JUHjD2_4x.jpg"
    evaluator = ImageQualityEvaluator(
        image_path=image_path,
        blur_threshold=100.0,
        noise_threshold=20.0,
        entropy_threshold=4.0,
        brisque_threshold=50.0,
        niqe_threshold=5.0,
        model_path=None  # 모델경로 지정 가능
    )

    try:
        result = evaluator.evaluate()
        evaluator.print_evaluation_results(result, image_name="Single Image")
    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except ValueError as e:
        print(e)
