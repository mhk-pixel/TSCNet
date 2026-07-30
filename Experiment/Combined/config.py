import torch
def config():
    return {
    # Paths
    'dataset_path': "/teamspace/studios/this_studio/phdResearch/data12/",
    
    'models_path': "/teamspace/studios/this_studio/Phd/Models",
    'output_dir': "/teamspace/studios/this_studio/Phd/experiment/Combined/results",
    #Cross Dataset if on
    'CrossDataset': "off", 
    'traindataset': "",
    'testdataset': "",
    # Device settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    
    # Model settings
    #Ablaition 
    'model_choice': 3 , #   0 , 1 , 2 , 3
    'model_params': {
        'n_channels': 19,
        'n_time_points': 640,
        'n_classes': 2,
        'dropout_rate': 0.3
      
    },
    
    # Cross-validation settings
    'k_folds': 5,
    'random_state': 42,
    'fold_strategy': 'manual132', #'stratified'
    'strategy': 'crossvalidation',
    'val_split': 0.0,  # No validation 
    'stratify_columns': ['Group'], 
    
    # Data settings
    'groups': ['HC', 'AD'], 
    'class_names': ['HC', 'AD'], 
    'segments_per_subject': None , # , # None
    'normalization': 'winsor_zscore', #winsor_zscore   minmax
    'norm_level': 'per_channel', # per_channel per_sample
    
    # Training parameters
    'params': {
        'batch_size': 16,
        'learning_rate': 0.0001 , 
        'epochs':30,
        'patience': 10
    },
    
    # Evaluation settings
    'evaluation': {
        'aggregation_mode': 'majority', #'average', 'majority'
        'confidence_filter': 0.70,
        'save_segment_csv': 'segment_predictions.csv',
        'validate_like_test': False
    },
    
    # Model saving settings
    'save_models': True,
    'save_best_only': True,
    'save_all_folds': False,
    
    # Experiment tracking
    'experiment_name': "ApplaitionM3", #None
    'save_config': True
}





