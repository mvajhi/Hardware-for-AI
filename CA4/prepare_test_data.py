import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def prepare_test_data(num_samples=100, output_file="test_data.txt"):
    """
    آماده‌سازی داده‌های تست برای مدل C
    """
    # تنظیمات مشابه با training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # بارگذاری داده‌های تست CIFAR-10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                           shuffle=False, num_workers=2)
    
    # ذخیره داده‌ها در فایل متنی
    with open(output_file, 'w') as f:
        count = 0
        for data in testloader:
            if count >= num_samples:
                break
                
            images, labels = data
            
            # تبدیل به numpy و flatten
            image = images.numpy().flatten()
            label = labels.item()
            
            # نوشتن label
            f.write(f"{label}")
            
            # نوشتن pixel values
            for pixel in image:
                f.write(f" {pixel:.6f}")
            f.write("\n")
            
            count += 1
    
    print(f"آماده‌سازی {num_samples} نمونه تست در {output_file} کامل شد")

if __name__ == "__main__":
    prepare_test_data(100, "test_data.txt")
