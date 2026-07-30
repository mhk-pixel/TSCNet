def folds():
    """
    Returns a dictionary mapping fold names to sets of subject IDs based on manual assignment.
    """
    return {
        'fold3': {
            'Sub_013', 'Sub_014', 'Sub_018', 'Sub_020',  'Sub_022', 'Sub_024','Sub_032', 'Sub_034', 
            'Sub_039', 'Sub_049', 'Sub_053', 'Sub_059', 'Sub_060'
        },
        'fold5': {
            'Sub_002', 'Sub_008', 'Sub_009', 'Sub_011',  'Sub_019','Sub_028', 'Sub_035', 'Sub_037', 
            'Sub_038', 'Sub_041', 'Sub_048',  'Sub_056', 'Sub_057'
        },
        'fold2': {
            'Sub_005', 'Sub_007', 'Sub_010', 'Sub_015', 'Sub_017', 'Sub_030', 'Sub_033', 'Sub_043', 
            'Sub_044', 'Sub_047', 'Sub_050', 'Sub_063', 'Sub_065'
        },
        'fold4': {
            'Sub_003', 'Sub_021',  'Sub_026', 'Sub_027', 'Sub_029', 'Sub_031', 'Sub_036', 'Sub_042', 
            'Sub_054', 'Sub_055', 'Sub_061', 'Sub_062', 'Sub_064'
        },
        'fold1': {
            'Sub_001', 'Sub_004','Sub_006','Sub_012',  'Sub_016', 'Sub_023', 'Sub_025', 'Sub_040', 
             'Sub_046','Sub_045', 'Sub_051', 'Sub_052', 'Sub_058'
        }
    }


