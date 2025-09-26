import os
from shutil import copyfile, rmtree
from tqdm import tqdm


if __name__ == '__main__':
    
    if not os.path.exists('datasets/source/archive.zip'):
        print("FAILED: The CUB200 data path datasets/source/archive.zip does NOT EXIST")
    else:
        print('Unzip archive.zip ...')
        os.system('unzip -qq datasets/source/archive.zip -d datasets/source/')
        
        with open('datasets/source/CUB_200_2011/images.txt') as f:
            img_names = f.readlines()
            
        with open('datasets/source/CUB_200_2011/train_test_split.txt') as f:
            splits = f.readlines()
        
        print('Processing the data ...')
        for img_name, split in tqdm(zip(img_names, splits), total=len(img_names)):
            idx1, filename = img_name.split(' ')
            filename = filename[:-1]
            
            idx2, is_train = split.split(' ')
            is_train = is_train[:-1]
            
            assert(idx1 == idx2)
            
            train_dir = os.path.join('datasets/source/CUB200/train', os.path.dirname(filename))
            val_dir = os.path.join('datasets/source/CUB200/val', os.path.dirname(filename))
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            if is_train == '1':
                copyfile(os.path.join('datasets/source/CUB_200_2011/images', filename), os.path.join(train_dir, os.path.basename(filename)))
            else:
                copyfile(os.path.join('datasets/source/CUB_200_2011/images', filename), os.path.join(val_dir, os.path.basename(filename)))
        
        
        if os.path.exists('datasets/source/CUB_200_2011'):
            rmtree('datasets/source/CUB_200_2011')
            
        if os.path.exists('datasets/source/attributes.txt'):
            os.remove('datasets/source/attributes.txt')
            
        if os.path.exists('datasets/source/cvpr2016_cub'):
            rmtree('datasets/source/cvpr2016_cub')
            
        if os.path.exists('datasets/source/segmentations'):
            rmtree('datasets/source/segmentations')
            
        if os.path.exists('datasets/source/archive.zip'):
            os.remove('datasets/source/archive.zip')
        
        print('Done! archive.zip file is removed.')
