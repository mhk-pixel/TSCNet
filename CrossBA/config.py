import torch
def config():
    return {
    # Paths
    'dataset_path': "",
    
    'models_path': "/teamspace/studios/this_studio/Phd/Models",
    'output_dir': "/teamspace/studios/this_studio/Phd/experiment/CrossBA/results",
        
    #Cross Dataset if on
    'CrossDataset': "on", 
    'traindataset': "/teamspace/studios/this_studio/phdResearch/data2/",
    'testdataset': "/teamspace/studios/this_studio/phdResearch/data1/",
        
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
        'dropout_rate': 0.3
      
    },
    
    # Cross-validation settings
    'k_folds': 1,
    'random_state': 42,
    'fold_strategy': "",
    'strategy': "",
    'val_split': 0.0,  # No validation 
    'stratify_columns': ['Group'], 
    
    # Data settings
    'groups': ['HC', 'AD'], 
    'class_names': ['HC', 'AD'], 
    'segments_per_subject': None , # , # None
    'normalization': 'minmax', 
    'norm_level': 'per_sample', 
    
    # Training parameters
    'params': {
        'batch_size': 16,
        'learning_rate': 0.001 ,
        'epochs':6,
        'patience': 10
    },
    
    # Evaluation settings
    'evaluation': {
        'aggregation_mode': 'majority', #'average', 'majority'
        'confidence_filter': 0.7,
        'save_segment_csv': 'segment_predictions.csv',
        'validate_like_test': False
    },
    
    # Model saving settings
    'save_models': True,
    'save_best_only': True,
    'save_all_folds': False,
    
    # Experiment tracking
    'experiment_name': "CrossBAGitHub", #None
    'save_config': True
}





