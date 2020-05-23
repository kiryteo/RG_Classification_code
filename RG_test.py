import glob
import argparse
from fastai import *
from fastai.vision import *
# In case testing with completely new unclassified samples

def model_test(mode, test_dir_path):
    """Function to test new samples"""
    if mode == 'version-1':
        model = load_learner('model_pickle/', test=ImageList.from_folder(test_dir_path))
    else:
        model = load_learner('augmented_model_pickle/', test=ImageList.from_folder(test_dir_path))

    # assumes test_dir_path to be without '/*' at the end, e.g. '~/RG_classification/test'
    test_dir_files = glob.glob(test_dir_path + '/*')

    Pred_FRI, Pred_FRII, Pred_Bent, Pred_Compact = 0

    for each_sample in test_dir_files:
        sample_img = open_image(each_sample)
        pred_class, pred_id, outputs = model.predict(sample_img)
        if str(pred_class) == 'FRI':
            Pred_FRI += 1
        elif str(pred_class) == 'FRII':
            Pred_FRII += 1
        elif str(pred_class) == 'Bent':
            Pred_Bent += 1
        else:
            Pred_Compact += 1

    print('Sample predictions: \n')
    print('Bent samples: ' + Pred_Bent)
    print('Compact samples: ' + Pred_Compact)
    print('FRI samples: ' + Pred_FRI)
    print('FRII samples: ' + Pred_FRII)

if __name__ == '__main__':
    print("Testing model on new samples")

    parser = argparse.ArgumentParser(description='Provide version of model (1/2) \
        to be used and the path to test directory')
    parser.add_argument('mode', type=str, help='Specify for using version-1 [1] \
        or version-2 [2] of the trained model')
    parser.add_argument('path', type=str, help='Provide path to test directory')

    args = parser.parse_args()

    if args.mode == '1':
        model_test('version-1', args.path)
    else:
        model_test('version-2', args.path)