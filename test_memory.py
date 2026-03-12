import torch
import sys
sys.path.append('..')

from model import MobileNetV2

def test_memory():
    '''

    :return:
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device}')

    if torch.cuda.is_available():
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}')
        print(f'Can use: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB \n')


    model = MobileNetV2(pretrained=True).to(device)
    model.eval()

    batch_sizes = [1, 2, 4, 8]
    img_size = 256

    print(f'Testing model params')
    for bs in batch_sizes:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            x = torch.randn(bs, 3, img_size, img_size).to(device)

            with torch.no_grad():
                y = model(x)

            if torch.cuda.is_available():
                mem_used = torch.cuda.max_memory_allocated()/1e6
                status = 'ok' if mem_used < 7000 else 'a lot of memory been used'
                print(f'{bs:<12} {mem_used:<15.2f}{status}')
            else:
                print(f'{bs:<12} {'N/A (CPU)' :<15}')

            del x, y
        except RuntimeError as e:
            print(f'{bs:<12} {"OOM":<15} {str(e)[:50]}')

if __name__ == '__main__':
    test_memory()