from omegaconf import OmegaConf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#https://github.com/vadimtimakin/Kaggle-Sign-Recognition/blob/main/config.py

config = {
    'general': {
        'experiment_name': 'baseline',
        'seed': 0xFACED,
        'num_classes': 2, 
    },

    'paths': {
        'path_to_train_data': BASE_DIR / 'data' / 'train.csv',
        'path_to_test_data': BASE_DIR / 'data' / 'test.csv',
        'path_to_project': BASE_DIR,
        'path_to_checkpoints': BASE_DIR / 'checkpoints' ,
        'path_to_submission': BASE_DIR / 'checkpoints' / 'submission', 
        'path_to_leaderboard': BASE_DIR / 'checkpoints' / 'leaderboard.csv',
        'path_to_NN_model': BASE_DIR / 'checkpoints' / 'DNN_models',
        'path_to_classic_model': BASE_DIR / 'checkpoints' / 'Classic_models',
    },
    
    'training': {
        # --- Общие ---
        'model_name': 'DNN',  # from src.pipeline: 'DNN' 'RFC' 'LogR' 'LogR_l1' 'LogR_elasticnet' 'KNN' 'SVC' 'CatBC' 'GradBC' 'LGBMClf'
        'test_size': 0.2,
        'scoring': 'accuracy',

        # --- Search / CV ---
        'search_verbose': 3,
        'search_n_iter': 50,
        'search_n_jobs': -1,

        # --- DNN: architecture ---
        'num_layers': 2, 
        'output_size': 2,
        'p_dropout': 0.5,

        # --- DNN: optimization ---
        'batch_size': 128,
        'num_epochs': 100,
        'lr': 0.01,
        'weight_decay': 0,

        # --- DNN: scheduler / early stopping ---
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
        'early_stopping_epochs': 100,

        # --- DNN: advanced training ---
        'mixed_precision': True,
        'gradient_accumulation': False,
        'gradient_accumulation_steps': 8,
        'gradient_clipping': False,
        'clip_value': 2,

        # --- DNN: warmup ---
        'warmup_scheduler': True,
        'warmup_epochs': 5,
        'warmup_multiplier': 100,

        # --- Debug ---
        'debug': False,
        'number_of_train_debug_samples': 5000,
        'number_of_val_debug_samples': 1000,

        # --- Saving ---
        'save_best': True,
        'save_last': False,
    },
}

config = OmegaConf.create(config)