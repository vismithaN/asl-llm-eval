import base64
from pathlib import Path
import random
import pandas as pd
from typing import List

class ASLDataPreparation:
    '''
    Create a dataframe with the following columns
    [base64 encoded image, label]
    Randomly select 30 images from each classes which results in 30*26=780 images
    Using the openai api to encode the image to base64
    '''
    def __init__(self, dataset_dir , images_per_class = 30):
        self.DATASET_DIR = Path(dataset_dir)
        self.LETTERS: List[str] = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.IMAGES_PER_CLASS = images_per_class
        self.train_path = self.DATASET_DIR / 'train'
        self.test_path = self.DATASET_DIR / 'test'
    
    def gather_letter_images(self, root: Path, letter: str) -> List[Path]:
        '''Return list of image Paths for a given letter from common locations'''
        candidates: List[Path] = []

        # train/test structure
        split_dir = root / 'train' / letter
        if split_dir.exists():
            candidates.extend(
                [p for p in split_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            )
        else:
            print("No directory found",split_dir)
        sample = random.sample(candidates, self.IMAGES_PER_CLASS)
        return sample


    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_image

    def build_base64_dataset(self) -> pd.DataFrame:
        """
        Build a DataFrame with columns [image_base64, label], sampling up to
        images_per_class images per ASL letter.
        """
        rows = []
        for letter in self.LETTERS:
            images = self.gather_letter_images(self.DATASET_DIR, letter)
            print(f'Found {len(images)} images for letter {letter}')
            if not images:
                print(f'Warning: No images found for letter {letter}')
                continue

            for file_path in images:
                try:
                    img_b64 = self.encode_image(file_path)
                    rows.append({'image_base64': img_b64, 'label': letter})
                except Exception as e:
                    print(f'Failed to encode {file_path}: {e}')

        df = pd.DataFrame(rows, columns=['image_base64', 'label'])
        print(f'Built dataset with {len(df)} rows (target ~ {self.IMAGES_PER_CLASS * len(self.LETTERS)}).')
        return df

    def save_dataset(self, df: pd.DataFrame, file_name: str):
        df.to_csv(file_name, index=False)

    def load_dataset(self, file_name: str) -> pd.DataFrame:
        return pd.read_csv(file_name)

    def data_analysis(self, df: pd.DataFrame):
        print(f"shape",df.shape)
        print(f"counts",df['label'].value_counts())
        print(f"unique",df['label'].unique())
        print(f"missing",df['label'].isnull().sum())
        print(f"Head",df.head(10))

