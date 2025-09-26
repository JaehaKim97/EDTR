import os

if __name__ == '__main__':
    os.makedirs('weights', exist_ok=True)
    os.makedirs('datasets/source', exist_ok=True)
    
    os.makedirs('experiments/cls/cub200', exist_ok=True)
    os.makedirs('experiments/seg/voc2012', exist_ok=True)
    os.makedirs('experiments/det/voc2012', exist_ok=True)
    
    # Automatically download StableDiffusion v2.1 weight
    file_path = os.path.join("weights", "v2-1_512-ema-pruned.ckpt")
    if os.path.exists(file_path):
        print(f"StableDiffusion v2.1 weight already exists: {file_path}")
    else:
        print("NOTE: StableDiffusion v2.1 weight not found. Attempting to download...")
        
        url = "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt"
        cmd = f"wget {url} --no-check-certificate -O {file_path}"
        exit_code = os.system(cmd)
        
        if exit_code == 0:
            print(f"Download completed successfully")
        else:
            print(f"Error: wget failed.")
        
    print("Setup Done:)")
