def folds():
    """
    Returns a dictionary mapping fold names to sets of subject IDs based on manual assignment.
    """
    return {
        'fold3': {
            'Sub_013', 'Sub_014', 'Sub_018', 'Sub_020',  'Sub_022', 'Sub_024','Sub_032', 'Sub_034', 
            'Sub_039', 'Sub_049', 'Sub_053', 'Sub_059', 'Sub_060',
            'Sub_078', 'Sub_079', 'Sub_082', 'Sub_084', 'Sub_085', 'Sub_087', 'Sub_095',
            'Sub_103', 'Sub_105', 'Sub_113', 'Sub_117', 'Sub_121', 'Sub_123', 'Sub_124'
            
        },
        'fold5': {
            'Sub_002', 'Sub_008', 'Sub_009', 'Sub_011',  'Sub_019','Sub_028', 'Sub_035', 'Sub_037', 
            'Sub_038', 'Sub_041', 'Sub_048',  'Sub_056', 'Sub_057',
            'Sub_067', 'Sub_073', 'Sub_074', 'Sub_076', 'Sub_083', 'Sub_091', 'Sub_098', 'Sub_101', 
            'Sub_102', 'Sub_108', 'Sub_112', 'Sub_114', 'Sub_120', 'Sub_128'
        },
        'fold2': {
            'Sub_005', 'Sub_007', 'Sub_010', 'Sub_015', 'Sub_017', 'Sub_030', 'Sub_033', 'Sub_043', 
            'Sub_044', 'Sub_047', 'Sub_050', 'Sub_063', 'Sub_065',
            'Sub_070', 'Sub_072', 'Sub_075', 'Sub_080', 'Sub_093', 'Sub_096', 'Sub_099', 'Sub_107',
            'Sub_111', 'Sub_118', 'Sub_126', 'Sub_129', 'Sub_131'
        },
        'fold4': {
            'Sub_003', 'Sub_021',  'Sub_026', 'Sub_027', 'Sub_029', 'Sub_031', 'Sub_036', 'Sub_042', 
            'Sub_054', 'Sub_055', 'Sub_061', 'Sub_062', 'Sub_064',
            'Sub_066', 'Sub_068', 'Sub_069', 'Sub_081', 'Sub_089', 'Sub_092', 'Sub_094', 'Sub_104', 
            'Sub_106', 'Sub_116', 'Sub_119', 'Sub_125', 'Sub_127'
        },
        'fold1': {
            'Sub_001', 'Sub_004','Sub_006','Sub_012',  'Sub_016', 'Sub_023', 'Sub_025', 'Sub_040', 
             'Sub_046','Sub_045', 'Sub_051', 'Sub_052', 'Sub_058',
            'Sub_071', 'Sub_077', 'Sub_086', 'Sub_088', 'Sub_090', 'Sub_097', 'Sub_100', 'Sub_109', 
            'Sub_110', 'Sub_115', 'Sub_122', 'Sub_130', 'Sub_132'
        }
    }


