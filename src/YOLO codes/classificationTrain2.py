import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class CustomizedDataset(ClassificationDataset):


    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        super().__init__(root, args, augment, prefix)

      
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz)),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):


    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):

    def build_dataset(self, img_path: str, mode: str = "train"):
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


model = YOLO("yolo11n-cls.pt")
model.train(data="data.yaml", trainer=CustomizedTrainer, epochs=100, imgsz=224, batch=64)
model.val(data="data.yaml", validator=CustomizedValidator, imgsz=224, batch=64)
