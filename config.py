from pathlib import Path
import torch

class Config:
    INPUT_DIR = Path('/mnt/d/ml/vk_practicum/picture _eng/data/INPUT_IMAGES')
    GT_DIR = Path('/mnt/d/ml/vk_practicum/picture _eng/data/GT_IMAGES')

    VAL_INPUT_PATH = Path('/mnt/d/ml/vk_practicum/picture _eng/test/INPUT_IMAGES')
    VAL_GT_PATH = Path('/mnt/d/ml/vk_practicum/picture _eng/test/expert_a_testing')

    # input params
    IMG_SIZE = 256
    IN_CHANNEL = 3
    OUT_CHANNEL = 3
    MODEL_NAME = 'mobilenetv2_unet'

    # training
    BATCH_SIZE = 2
    NUM_EPOCHS_FREEZE = 10
    NUM_EPOCHS_UNFREEZE = 50
    LR = 1e-4
    LR_UF = 1e-5
    NUM_WORKERS = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    #gpu memory optimize
    USE_MIXED_PRECISION = True
    GRADIENT_ACCUMULATION = 1 # увеличивает если batch_size маленький
    PIN_MEMORY = True # ускорение передачи CPU-GPU

    #defects
    DEFECTS_NAME = ['0', 'N1', 'N1.5', 'P1', 'P1.5']

    #Checkpoints
    CHECKPOINT_DIR = Path('checkpoint')
    MODEL_DIR = Path('models')

    @classmethod
    def setup_dirs(cls):
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_gpu_info(cls):
        '''
        information about gpu
        '''
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
            print(f'{torch.cuda.get_device_capability(0)}')
        else:
            print('GPU is not available')
            