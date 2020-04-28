class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Path/to/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/path/to/VAR/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = './dataloaders/hmdb51'

            output_dir = './dataloaders/hmdb51_processed'

            return root_dir, output_dir
        elif database == 'kaggle':
            # folder that contains class labels
            root_dir = '../Downloads/deepfake-detection-challenge/train'

            output_dir = './dataloaders/deepfake-processed'

            return root_dir, output_dir
        elif database == 'celeb-df':
            # folder that contains class labels
            root_dir = '../Downloads/Celeb-df'

            output_dir = '../Downloads/Celeb-DF-v2-I3D-processed-cropped'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './c3d-pretrained.pth'