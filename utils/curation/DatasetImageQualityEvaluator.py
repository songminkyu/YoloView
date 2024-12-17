import cv2
import numpy as np
import torch
import torch.nn as nn
import pyiqa
import shutil
import os
from torchvision import transforms, models
from typing import Tuple, Dict
from utils.brisque.brisque import BRISQUE

class ImageLoader:
    """이미지를 로드하고 유효성을 검사하는 클래스."""

    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image at path {image_path} not found.")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}.")
        return image

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

    def calculate_iqa_brisque(self, image: np.ndarray) -> Tuple[bool, float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # GPU가 없는 경우 CPU로 fallback
        im_t = torch.FloatTensor(image.transpose(2, 0, 1)[None]) / 255.0
        im_t = im_t.to(device)

        try:
            brisque_metric = pyiqa.create_metric("brisque", device=device)
            score_tensor = brisque_metric(im_t)
            score = score_tensor.item()
        except Exception as e:
            raise RuntimeError(f"Brisque calculation failed: {e}")

        is_good = score < self.threshold
        return is_good, score


    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.calculate_brisque(image)

class PIQECalculator:
    """PIQE 점수 계산 클래스."""

    def __init__(self, image_path: str, threshold: float = 5.0):
        """
        Args:
            threshold (float): NIQE 임계값. 점수가 이 값보다 작을수록 품질이 좋다고 판단.
        """
        self.threshold = threshold
        self.image_path = image_path

    def calculate_piqe(self, image: np.ndarray) -> Tuple[bool, float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # GPU가 없는 경우 CPU로 fallback
        im_t = torch.FloatTensor(image.transpose(2, 0, 1)[None]) / 255.0
        im_t = im_t.to(device)

        try:
            piqe_metric = pyiqa.create_metric("piqe", device=device)
            score_tensor = piqe_metric(im_t)
            score = score_tensor.item()
        except Exception as e:
            raise RuntimeError(f"PIQE calculation failed: {e}")

        is_good = score < self.threshold
        return is_good, score

    def evaluate(self) -> Tuple[bool, float]:
        loader = ImageLoader()
        image = loader.load_image(self.image_path)
        return self.calculate_piqe(image)


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


class ImageQualityAssessmentReorganizer:
    """
    품질 평가 결과에 따라 이미지/라벨을 재배치하는 클래스.
    dataset_splits 없이 src_root_dir 아래 train/test/valid 구조를 직접 순회한다.

    예)
    src_root_dir/
      train/
        images/
        labels/
      test/
        images/
        labels/
      valid/
        images/
        labels/

    를

    dest_root_dir/
      BRISQUE_{src_root_basename}/
        train/
          images/
          labels/
        test/
          images/
          labels/
        valid/
          images/
          labels/

    형태로 재배치.
    """

    def __init__(self, src_root_dir: str, dest_root_dir: str, metric_name: str, threshold: float = 50.0):
        self.src_root_dir = os.path.abspath(src_root_dir)
        self.dest_root_dir = dest_root_dir
        self.metric_name = metric_name
        self.threshold = threshold

        # 예: dest_root_dir/BRISQUE_srcdir
        self.metric_folder = os.path.join(dest_root_dir, f"{metric_name}_{os.path.basename(self.src_root_dir)}")
        os.makedirs(self.metric_folder, exist_ok=True)

    def move_files_by_metric(self):
        splits = ["train", "test", "valid"]
        for split in splits:
            src_images_dir = os.path.join(self.src_root_dir, split, "images")
            src_labels_dir = os.path.join(self.src_root_dir, split, "labels")

            # 대상 경로 생성
            dest_images_dir = os.path.join(self.metric_folder, split, "images")
            dest_labels_dir = os.path.join(self.metric_folder, split, "labels")
            os.makedirs(dest_images_dir, exist_ok=True)
            os.makedirs(dest_labels_dir, exist_ok=True)

            if not os.path.exists(src_images_dir):
                print(f"Source images directory does not exist: {src_images_dir}")
                continue

            # 이미지 파일 반복
            for file_name in os.listdir(src_images_dir):
                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(src_images_dir, file_name)
                    label_name = os.path.splitext(file_name)[0] + ".txt"
                    label_path = os.path.join(src_labels_dir, label_name)

                    # BRISQUE 평가
                    is_good = False
                    score = 0.0

                    if self.metric_name == "BRISQUE":
                        brisque_calculator = BRISQUECalculator(img_path, threshold=self.threshold)
                        is_good, score = brisque_calculator.evaluate()
                    elif self.metric_name == "NIQE":
                        niqe_calculator = NIQECalculator(img_path, threshold=self.threshold)
                        is_good, score = niqe_calculator.evaluate()
                    elif self.metric_name == "PIQE":
                        piqe_calculator = PIQECalculator(img_path, threshold=self.threshold)
                        is_good, score = piqe_calculator.evaluate()
                    else:
                        ent_calculator = EntropyCalculator(img_path, threshold=self.threshold)
                        is_good, score = ent_calculator.evaluate()

                    if is_good == False:
                        # 결과에 따라 동일한 구조로 이동(여기서는 is_good 여부 상관없이 이동)
                        dest_img_path = os.path.join(dest_images_dir, file_name)
                        shutil.move(img_path, dest_img_path)

                        if os.path.exists(label_path):
                            dest_label_path = os.path.join(dest_labels_dir, label_name)
                            shutil.move(label_path, dest_label_path)
                        else:
                            print(f"Label file not found for {img_path}")
                    else:
                        print(f"Bad image: {img_path} (Score: {score})")


class ImageQualityEvaluator:
    """이미지 품질 평가를 종합하는 클래스."""

    def __init__(self,
                 image_path : str,
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
    src_root_dir = "c:\\Users\\USER\\Downloads\\rock-paper-scissors.v14i.yolov11"
    dest_root_dir = "c:\\Users\\USER\\Downloads\\rock-paper-scissors.v14i.yolov11"
    metric_name = "PIQE"

    reorganizer = ImageQualityAssessmentReorganizer(
        src_root_dir=src_root_dir,
        dest_root_dir=dest_root_dir,
        metric_name=metric_name,
        threshold=5.0
    )

    reorganizer.move_files_by_metric()
    '''
     image_path = "c:\\Users\\USER\\Desktop\\M1JUHjD2_4x.jpg"
    evaluator = ImageQualityEvaluator(
        image_path=image_path,
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
    '''

