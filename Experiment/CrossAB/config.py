import torch
def config():
    return {
    # Paths
    'dataset_path': "",
    
    'models_path': "/teamspace/studios/this_studio/Phd/Models",
    'output_dir': "/teamspace/studios/this_studio/Phd/experiment/CrossAB/results",
        
    #Cross Dataset if on
    'CrossDataset': "on", 
    'traindataset': "/teamspace/studios/this_studio/phdResearch/data1/",
    'testdataset': "/teamspace/studios/this_studio/phdResearch/data2/",
   
        
    # Device settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    
    # Model settings
    #Ablaition 
    'model_choice': 0 , #   0 , 1 , 2 , 3
    'model_params': {
        'n_channels': 19,
        'n_time_points': 640,
        'n_classes': 2,
        'dropout_rate': 0.2
      
    },
    
    # Cross-validation settings
    'k_folds': 1,
    'random_state': 42,
    'fold_strategy': "manual",
    'strategy': 'crossvalidation',
    'val_split': 0.0,  # No validation 
    'stratify_columns': ['Group'], 
    
    # Data settings
    'groups': ['HC', 'AD'], 
    'class_names': ['HC', 'AD'], 
    'segments_per_subject': None , # , # None
    'normalization': 'none', #winsor_zscore   minmax
    'norm_level': 'per_channel', # per_channel per_sample
    
    # Training parameters
    'params': {
        'batch_size': 16,
        'learning_rate': 0.000001 , #0.00001,
        'epochs':7,
        'patience': 10
    },
    
    # Evaluation settings
    'evaluation': {
        'aggregation_mode': 'majority', #'average', 'majority'
        'confidence_filter': 0.3,
        'save_segment_csv': 'segment_predictions.csv',
        'validate_like_test': False
    },
    
    # Model saving settings
    'save_models': True,
    'save_best_only': True,
    'save_all_folds': False,
    
    # Experiment tracking
    'experiment_name': "CrossABGitHub", #None
    'save_config': True
}





