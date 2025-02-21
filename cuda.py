import torch
import gc

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available!")
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))
        print(f"Environment CUDA verison: {torch.version.cuda}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"Current CUDA device count: {torch.cuda.device_count()}")
        torch.cuda.empty_cache()
        gc.collect()

    else:
        print("CUDA is not available. Running on CPU.")

if __name__ == "__main__":
    main()