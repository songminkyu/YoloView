import cv2
import numpy as np
import torch
import torch.nn as nn
import pyiqa
from torchvision import transforms, models
from typing import Tuple, Dict, List
from utils.brisque.brisque import BRISQUE

class ImageQualityEvaluator:
    def __init__(self,
                 blur_threshold=100.0,
                 noise_threshold=20.0,
                 entropy_threshold=4.0,
                 brisque_threshold=50.0,
                 niqe_threshold=0.5,
                 model_path=None):
        """
        이미지 품질 평가를 위한 초기화.
        Args:
            blur_threshold (float): 블러 감지 임계값
            noise_threshold (float): 노이즈 감지 임계값
            entropy_threshold (float): 엔트로피 임계값
            brisque_threshold (float): BRISQUE 임계값
            niqe_threshold (float) : niqe 임계값
            model_path (str): 딥러닝 모델 경로 (품질 예측)
        """
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold
        self.entropy_threshold = entropy_threshold
        self.brisque_threshold = brisque_threshold
        self.niqe_threshold = niqe_threshold
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

    def calculate_entropy(self, image: np.ndarray) -> Tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_prob = hist / hist.sum()
        entropy = -np.sum([p * np.log2(p) for p in hist_prob if p > 0])
        # 예: 엔트로피가 일정 값 이상이면 (분포가 다양) 품질 좋다고 볼 수도 있음 (임의 기준)
        is_good = entropy > self.entropy_threshold
        return is_good, entropy

    def calculate_brisque(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        BRISQUE 품질 점수 계산 (낮을수록 품질이 좋음).
        낮은 점수일수록 품질이 좋으므로, 임계값보다 작으면 True
        """
        brisque_score = BRISQUE(url=False)
        score = brisque_score.score(image)
        is_good = score < self.brisque_threshold
        return is_good, score

    def calculate_niqe(self, image: np.ndarray) -> Tuple[bool, float]:
        # 이미지를 [N, C, H, W] 형식으로 변환하고 [0, 1] 범위로 정규화
        # image: (H, W, C)
        im_t = torch.FloatTensor(image.transpose(2, 0, 1)[None]).to('cuda') / 255.0

        # NIQE metric 객체 생성 (GPU 사용)
        niqe_metric = pyiqa.create_metric("niqe", device='cuda')

        # NIQE 점수 계산
        score_tensor = niqe_metric(im_t)
        score = score_tensor.item()

        # threshold를 기준으로 품질 판단 (NIQE 점수가 낮을수록 품질이 좋다고 판단)
        is_good = score < self.niqe_threshold

        return is_good, score

    def predict_quality_with_model(self, image: np.ndarray) -> float:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            quality_score = self.quality_model(image_tensor).item()
        return quality_score



    def evaluate(self, image_path: str) -> Dict[str, Tuple[bool, float]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")

        results = {}
        results["blur"] = self.detect_blur(image)
        results["noise"] = self.detect_noise(image)
        results["niqe"] = self.calculate_niqe(image)
        results["entropy"] = self.calculate_entropy(image)
        results["brisque"] = self.calculate_brisque(image)

        if self.quality_model:
            dl_quality_score = self.predict_quality_with_model(image)
            # 예를 들어, dl_quality_score가 클수록 좋다고 가정하고 threshold 기준 설정
            dl_threshold = 0.5
            is_good = dl_quality_score > dl_threshold
            results["dl_quality"] = (is_good, dl_quality_score)

        return results

# 출력 함수
def print_evaluation_results(results: Dict[str, Tuple[bool, float]], image_name: str = "Image"):
    print(f"{image_name} Quality Evaluation Results:")
    for key, (status, metric) in results.items():
        print(f"  {key.capitalize()}: {status} (Score: {metric:.2f})")
    print()

# 실행 예시
if __name__ == "__main__":
    evaluator = ImageQualityEvaluator(
        blur_threshold=100.0,
        noise_threshold=20.0,
        entropy_threshold=4.0,
        brisque_threshold=50.0,
        niqe_threshold=5.0
    )

    image_path = "c:\\Users\\USER\\Desktop\\M1JUHjD2_4x.jpg"
    try:
        result = evaluator.evaluate(image_path)
        print_evaluation_results(result, image_name="Single Image")
    except FileNotFoundError as e:
        print(e)
