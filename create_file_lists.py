import os
from tqdm import tqdm


# base_path = '/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Tue_151515_highPT/'
# base_path = '/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Mon_235233_highPT/'
base_path = '/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-tagging/highPT/Mon_115056_highPT/'
# train_file = 'all_samples.txt'
# test_file = 'dummy.txt'
train_file = 'train_val_samples.txt'
test_file = 'test_samples.txt'
# train_file = 'train_val_samples_balanced.txt'
# test_file = 'test_samples_balanced.txt'
# train_file = 'train_val_samples_enhanced.txt'
# test_file = 'test_samples_enhanced.txt'


if __name__ == '__main__':
    with open(train_file, 'w') as trainf_global, open(test_file, 'w') as testf_global:
        dir_list = os.listdir(base_path)
        for cur_path in tqdm(dir_list, total=len(dir_list)):
            cur_full_path = os.path.join(base_path, os.path.join(cur_path, 'output'))
            train_full_path = os.path.join(cur_full_path, train_file)
            if os.path.isfile(train_full_path):
                with open(train_full_path, 'r') as trainf:
                    for line in trainf.readlines():
                        trainf_global.write(cur_full_path+'/'+line)
            test_full_path = os.path.join(cur_full_path, test_file)
            if os.path.isfile(test_full_path):
                with open(test_full_path, 'r') as testf:
                    for line in testf.readlines():
                        testf_global.write(cur_full_path+'/'+line)