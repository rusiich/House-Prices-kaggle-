from omegaconf import OmegaConf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#https://github.com/vadimtimakin/Kaggle-Sign-Recognition/blob/main/config.py

config = {
    'general': {
        'experiment_name': 'v1',
        'seed': 0xFACED,
        'num_classes': 2, 
        
    },

    'paths': {
        'path_to_train_data': BASE_DIR / 'data' / 'train.csv',
        'path_to_test_data': BASE_DIR / 'data' / 'test.csv',
        'path_to_project': BASE_DIR,
        'path_to_checkpoints': BASE_DIR / 'checkpoints' ,
        'path_to_submission': BASE_DIR / 'checkpoints' / 'submission', 

    },
    
    'training': {
        'model_name': 'RFC' , #from src.pipeline 'DNN' 'RFC' 'KNN' 'LogR'
        'num_epochs': 10,
        'early_stopping_epochs': 100,
        'lr': 1e-4 / 100,

        'mixed_precision': True,
        'gradient_accumulation': False,
        'gradient_clipping': False,
        'gradient_accumulation_steps': 8,
        'clip_value': 2,

        'test_size':.2,
        'batch_size': 64,
        
        'warmup_scheduler': True,
        'warmup_epochs': 5,
        'warmup_multiplier': 100,

        'debug': False,
        'number_of_train_debug_samples': 5000,
        'number_of_val_debug_samples': 1000,
        
        # 'device':  ,
        'save_best': True,
        'save_last': False,
    },
}

config = OmegaConf.create(config)