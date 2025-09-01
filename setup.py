import os

if __name__ == '__main__':
    os.makedirs('weights', exist_ok=True)
    os.makedirs('datasets/source', exist_ok=True)
    
    os.makedirs('experiments/cls/cub200', exist_ok=True)
    os.makedirs('experiments/seg/voc2012', exist_ok=True)
    os.makedirs('experiments/det/voc2012', exist_ok=True)

    print("Setup Done:)")
