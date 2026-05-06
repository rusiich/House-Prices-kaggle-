from omegaconf import OmegaConf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#https://github.com/vadimtimakin/Kaggle-Sign-Recognition/blob/main/config.py

config = {
    'general': {
        'experiment_name': 'v4',
        'seed': 0xFACED,
        'num_classes': 1, 
    },
    # --- PATHS ---
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
        # from src.pipeline: 'All' 'Ensemble_AVG' 'Ensemble_voting' 'DNN' 
        #  'RandomForest' 'GradientBoosting' 'CatBoost' 'KNN'
        # 'Linear' 'Ridge'  'Lasso' 'ElasticNet'
        'model_name': 'Ensemble_AVG',  
        'test_size': 0.2,
        'scoring': 'neg_root_mean_squared_error',

        # --- Search / CV ---
        'search_verbose': 1,
        'search_n_iter': 50,
        'search_n_jobs': -1,

        # --- DNN: architecture ---
        'num_layers': 5, 
        'output_size': 1,
        'p_dropout': 0,

        # --- DNN: optimization ---
        'batch_size': 16,
        'num_epochs': 100,
        'lr': 0.01,
        'weight_decay': 1e-5,

        # --- DNN: scheduler / early stopping ---
        'scheduler_factor': 0.5,
        'scheduler_patience': 20,
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

    'logging': {
        'use_wandb': False,
        'wandb_project_name': 'titanic',
        'entity': None,
    }
}

config = OmegaConf.create(config)