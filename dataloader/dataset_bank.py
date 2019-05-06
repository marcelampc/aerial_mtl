from os import listdir
from os.path import join
from ipdb import set_trace as st
import glob
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_std(root, data_split, tasks):
    input_list = sorted(glob.glob(join(root, 'rgb', data_split, '*.png')))
    targets_list = []
    for task in tasks:
        targets_list.append(sorted(glob.glob(join(root, task, data_split, '*.png'))))
    # return list(zip(input_list, targets_list))
    return input_list, targets_list

def dataset_target_only(root, phase):
    return sorted(glob.glob(join(root, 'depth', phase, '*.png')))

def dataset_kitti(root, phase, opt):
    phase_path = join(root, phase)
    kitti_split = opt.kitti_split

    missing_file_list = open('missing_{}_{}'.format(phase, kitti_split), 'w')

    if kitti_split.find('rob') is not -1:
        if phase != 'test':
            if kitti_split.find('_r') is not -1:
                index = '[2]'
            else:
                index = '[2,3]'
            # search for RGB images
            image_search = "2011_*_*_drive_*_sync/image_0{}/*.png".format(index)
            image_files = sorted(glob.glob(join(phase_path, image_search)))

            # search for depth images
            depth_search = "2011_*_*_drive_*_sync/proj_depth/groundtruth/image_0{}/*.png".format(index)
            depth_files = sorted(glob.glob(join(phase_path, depth_search)))
        else:
            # search for RGB images
            image_search = "image/*.png"
            image_files = sorted(glob.glob(join(phase_path, image_search)))

            # limitation of my program
            depth_files = image_files  # correct it to not impose the need of depth images
    elif kitti_split.find('eigen') is not -1:
        image_files = []
        file_ = open("config/kitti/eigen_{}_files_rob.txt".format(phase), "r")

        for f in file_:
            phase_path = phase_path = join(root, phase)
            filenames = f.rsplit(' ', 1)
            if kitti_split.find('_r') is not -1:
                f = filenames[0]
                filepaths = sorted(glob.glob(join(root, '*', f.strip('\n'))))
                if filepaths:
                    image_files.append(filepaths[0])
            else:
                for f in filenames:
                    filepaths = sorted(glob.glob(join(root, '*', f.strip('\n'))))
                    if filepaths:
                        image_files.append(filepaths[0])

        # if phase == 'test':
        #     depth_files = image_files  # correct it to not impose the need of depth images
        # else:
        depth_files = [f.replace('sync/', 'sync/proj_depth/groundtruth/') for f in image_files]

    return list(zip(image_files, depth_files))


def dataset_oneraroom(root, phase):
    # phase_path = join(root, phase)

    filepath = "config/oneraroom/onera_{}_files.txt".format(phase)

    image_files = []
    depth_files = []

    with open(filepath, 'r') as f:
        for line in f:
            print('Folder: {} in phase: {}'.format(line, phase))
            image_search = join(root, line.strip('\n'), 'image', '*.jpg')
            depth_search = join(root, line.strip('\n'), 'depth', '*.png')

            [image_files.append(imagepath) for imagepath in sorted(glob.glob(image_search))]
            [depth_files.append(depthpath) for depthpath in sorted(glob.glob(depth_search))]

    return [(image_file, depth_file) for (image_file, depth_file) in zip(image_files, depth_files)]


def dataset_3drms_list(root, phase, index, ext=''):
    phase_path = join(root, phase)
    scene_folders = '*{}'.format(index)
    vcam_ids = '*[02468]'
    file_ext = '*{}'.format(ext)
    return sorted(glob.glob(join(phase_path, scene_folders, vcam_ids, file_ext)))


def dataset_3drms_(root, phase): # phase = [test,train,val]_split
    # This dataset does not have depth maps neither for validation, neither for tests
    if 'train' in phase:
        if '1' in phase:
            index = '[0][012][026][048]' # [0128,0160,0224]
        elif '2' in phase:
            index = '[0][01][026][018]' # [0001,0128,0160]
        else:  # train on all dataset
            index = ''
        return index
    elif 'val' in phase:
        if '1' in phase:
            index = '[0][0][0][1]'
        elif '2' in phase:
            index = '[0][2][2][4]'
        return index


def dataset_3drms(root, phase, use_semantics=False): # phase = [test,train,val]_split
    if 'test' in phase:
        phase='testing'
        phase_path = join(root, phase)
        image_files = sorted(glob.glob(join(phase_path, '*', '*', '*.png')))
        return list(zip(image_files, image_files))
    else:
        index = dataset_3drms_(root, phase)
        phase='training'
        image_list = dataset_3drms_list(root, phase, index, ext='undist.png')
        depth_list = dataset_3drms_list(root, phase, index, ext='.bin')
        if use_semantics:
            semantics_list = dataset_3drms_list(root, phase, index, ext='gtr.png')
            return list(zip(image_list, depth_list, semantics_list))
        return list(zip(image_list, depth_list))

def dataset_3drms_stereo(root, phase, use_semantics=False):
    if 'test' in phase:
        phase='testing'
        phase_path = join(root, phase)
        image_files = sorted(glob.glob(join(phase_path, '*', '*', '*.png')))
        return list(zip(image_files, image_files))
    else:
        index = dataset_3drms_(root, phase)
        phase='training'
        image_list = dataset_3drms_list(root, phase, index, ext='undist.png')
        depth_list = dataset_3drms_list(root, phase, index, ext='.bin')
        stereo_depth_list = dataset_3drms_list(root, phase='depth_from_stereo/sgbm_depth_map/training', index=index, ext='dmap.png')
        if use_semantics:
            semantics_list = dataset_3drms_list(root, phase, index, ext='gtr.png')
            return list(zip(image_list, depth_list, semantics_list, stereo_depth_list))
        return list(zip(image_list, depth_list, stereo_depth_list))

def dataset_dfc_get_semantics(root):
    return glob.glob(join(root, 'TrainingGT/', '*.tif'))

def dataset_dfc_get_heights(root, which_raster):
    label_file = glob.glob(join(root, 'LidarGeoTiffRasters/DSM_C12/', '*.tif'))
    if which_raster != 'dsm':
        dem_types = ('dsm_demb', 'dsm_dem3msr', 'dsm_demtli')
        dem_names = ('DEM+B_C123', 'DEM_C123_3msr', 'DEM_C123_TLI')
        for i, dem_type in enumerate(dem_types):
            if which_raster in dem_type:
                dem_name = dem_names[i]
                break
        print('DEM is {}'.format(dem_name))
        label_file_dem = glob.glob(join(root, 'LidarGeoTiffRasters/{}/'.format(dem_name), '*.tif'))
        return [label_file[0], label_file_dem[0]]
    else:
        return [label_file]

def dataset_dfc_get_rgb(root, data_split):
    if 'test' in data_split:
        # index = ['UH_NAD83_271460_3289689']
        index = [
                  'UH_NAD83_271460_3289689',
                 'UH_NAD83_271460_3290290',
                 'UH_NAD83_272056_3290290',
                 'UH_NAD83_272652_3290290',
                 'UH_NAD83_273248_3290290',
                 'UH_NAD83_273844_3290290',
                 'UH_NAD83_274440_3289689',
                 'UH_NAD83_274440_3290290',
                 'UH_NAD83_275036_3289689',
                 'UH_NAD83_275036_3290290',
                #  'UH_NAD83_272056_3289689', # train
                #  'UH_NAD83_272652_3289689',
                #  'UH_NAD83_273248_3289689',
                #  'UH_NAD83_273844_3289689'
                ]
    elif 'val' in data_split:
        index = ['UH_NAD83_272652_3289689']
    elif 'train' in data_split:
        index = [
                 'UH_NAD83_272056_3289689',
                 'UH_NAD83_272652_3289689',
                 'UH_NAD83_273248_3289689',
                 'UH_NAD83_273844_3289689'
                 ]
    return [join(root, 'FinalRGBHRImagery', idx + '.tif') for idx in index]


def dataset_dfc(root, data_split, phase, model=None, which_raster='dsm'):
    # when data_split is test and may need semantics, send None
    if 'semantics' in model:
        if 'test' in phase:
            return dataset_dfc_get_rgb(root, data_split)
        else:
            if 'test' in data_split:
                sys.exit("There is no semantic data for test split.")
            return dataset_dfc_get_rgb(root, data_split), dataset_dfc_get_semantics(root)
    elif 'regression' in model:
        return dataset_dfc_get_rgb(root, data_split), dataset_dfc_get_heights(root, which_raster)
    elif 'multitask' in model:
        if 'test' in data_split:
            return dataset_dfc_get_rgb(root, data_split), dataset_dfc_get_heights(root, which_raster)
        else:
            target = dataset_dfc_get_heights(root, which_raster)
            target.append(dataset_dfc_get_semantics(root)[0])
            return dataset_dfc_get_rgb(root, data_split), target

def dataset_nyu_deblur(root, root2, data_split):
    rgb_path = sorted(glob.glob(join(root, 'rgb', data_split, '*.png')))
    # Depth map
    target1_path = sorted(glob.glob(join(root, 'depth', data_split, '*.png')))
    # In focus correspondant to rgb
    target2_path = sorted(glob.glob(join(root2, 'rgb', data_split, '*.png')))
    return list(zip(rgb_path, target1_path, target2_path))

def get_idx_vaihingen(data_split):
    if 'test' in data_split:
        index = [
                'area4',
                'area6',
                'area8',
                'area10',
                'area12',
                'area14',
                'area16',
                'area2',
                'area20',
                'area22',
                'area24',
                'area27',
                'area29',
                'area31',
                'area33',
                'area35',
                'area38',
                ]
    elif 'val' in data_split: # ToDo: check if validate to get the right files!
        index = ['area4',
                'area6',
                'area8',
                'area10',
                ]
    elif 'train' in data_split:
        index = [
                'area1',
                'area3',
                'area5',
                'area7',
                'area11',
                'area13',
                'area15',
                'area17',
                'area21',
                'area23',
                'area26',
                'area28',
                'area30',
                'area32',
                'area34',
                'area37',
                ]
    return index

def dataset_vaihingen_get_rgb(root, data_split):
    index = get_idx_vaihingen(data_split)
    return [join(root, 'top', 'top_mosaic_09cm_' + idx + '.tif') for idx in index]
    
def dataset_vaihingen_get_semantics(root, data_split):
    index = get_idx_vaihingen(data_split)
    return [join(root, 'gts_for_participants', 'top_mosaic_09cm_' + idx + '.tif') for idx in index]

def dataset_vaihingen_get_heights(root, data_split):
    index = get_idx_vaihingen(data_split)
    index = [w.replace('top_mosaic_09cm', 'dsm') for w in index]
    # original DSM
    # return [join(root, 'dsm', 'dsm_09cm_matching_' +  idx + '.tif') for idx in index]

    # normalized DSM (cDSM)
    return [join(root, 'nDSM', 'dsm_09cm_matching_' +  idx + '_normalized.jpg') for idx in index]

def dataset_vaihingen(root, data_split, phase, model=None):
    if 'semantics' in model:
        if 'test' in phase:
            return dataset_vaihingen_get_rgb(root, data_split)
        else:
            if 'test' in data_split:
                sys.exit("There is no semantic data for test split.")
            return dataset_vaihingen_get_rgb(root, data_split), dataset_vaihingen_get_semantics(root, phase)
    elif 'regression' in model:
        return dataset_vaihingen_get_rgb(root, data_split), dataset_vaihingen_get_heights(root, data_split)
    elif 'multitask' in model:
        if 'test' in data_split:
            return dataset_vaihingen_get_rgb(root, data_split), dataset_vaihingen_get_heights(root, data_split)
        else:
            target = list(zip(dataset_vaihingen_get_heights(root, data_split), dataset_vaihingen_get_semantics(root, phase)))
            return dataset_vaihingen_get_rgb(root, data_split), target
