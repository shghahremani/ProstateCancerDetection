from kwae.dataset import conversion



opt={'imgDir': 'Original_Images_v3/',
    'maskDir': 'Masks_Seg_Labels_v3/',
    'rootDir': '/home/siavash/KroniKare/Data/Latest_orient/',
    'rows': 360,
    'cols': 480,
    'depth': 3,

       }
conversion.convert_wound_mask_to(opt, 'ImgSeg_OnlyWound')