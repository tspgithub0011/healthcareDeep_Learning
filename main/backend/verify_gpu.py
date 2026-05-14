import torch

def verify():
    print("-" * 30)
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test a simple tensor operation on GPU
        try:
            x = torch.rand(5, 3).cuda()
            print("Successfully moved tensor to GPU!")
        except Exception as e:
            print(f"Error testing GPU: {e}")
    else:
        print("GPU not detected. Using CPU.")
    print("-" * 30)

if __name__ == "__main__":
    verify()
