import os
import json
from tqdm import tqdm
from datasets import load_dataset
from datasets.features import Image as ImageFeature

from src.logger import logger
from src.entity.entity import DataIngestorConfig

class DataIngestor:
    def __init__(self, config: DataIngestorConfig):
        self.config=config
    
    def prepare(self):
        self.prepare_dataset()
        self.prepare_mapping()

    def prepare_dataset(self):
        columns=['image', 'label_bbox_enriched']
        dataset=load_dataset(self.config.DATASET_NAME).select_columns(columns).cast_column('image', ImageFeature(mode='RGB'))

        self.dataset=dataset.map(self.process_fn, batched=True, batch_size=1024, remove_columns='label_bbox_enriched')
        self.dataset.save_to_disk(self.config.ARTIFACT_DIR)

    @staticmethod
    def process_fn(batch):
        target = [x if x is not None else [] for x in batch['label_bbox_enriched']]
        return {'label_bbox': target}

    def prepare_mapping(self):
        labels=[]
        for split in ['train', 'test']:
            for sample in tqdm(self.dataset[split]):
                    labels.extend([box['label'] for box in sample['label_bbox']])

        labels=list(set(labels))
        label_to_id = {label:idx for idx, label in enumerate(labels)}
        id_to_label = {idx:label for idx, label in enumerate(labels)}

        with open(self.config.ARTIFACT_DIR+'/label_to_id.json', 'w') as file:
            json.dump(label_to_id, file, indent=4)
        with open(self.config.ARTIFACT_DIR+'/id_to_label.json', 'w') as file:
            json.dump(id_to_label, file, indent=4)
    
    def sync_to_s3(self):
        command = f'aws s3 sync {self.config.ARTIFACT_DIR} s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}'
        os.system(command)
        logger.info(f"Sync data from {self.config.ARTIFACT_DIR} to s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}")
    
    def sync_from_s3(self):
        command = f'aws s3 sync s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} {self.config.ARTIFACT_DIR}'
        os.system(command)
        logger.info(f"Sync data from s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} to {self.config.ARTIFACT_DIR}")