import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import time
from datetime import timedelta

import lime
from lime import lime_image
import shap
from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# -------------------------- VGG16(https://github.com/kuangliu/pytorch-cifar) --------------------------
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG16():
    return VGG('VGG16')

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training')):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc='Testing')):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    return test_loss, accuracy

def denormalize(img_tensor, device):
    """Inverse normalization tensor to PIL image (for visualization)"""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(device)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))

def get_topk_pixels(attr_map, drop_ratio=0.5):
    attr_flat = attr_map.flatten()
    k = int(len(attr_flat) * drop_ratio)
    # Sort by the absolute value of significance from small to large, and select the top k (most important pixels)
    topk_indices = np.argsort(np.abs(attr_flat))[-k:]
    #least important pixels
    #topk_indices = np.argsort(np.abs(attr_flat))[:k]
    # Convert to 2D coordinates (H, W)
    topk_coords = np.unravel_index(topk_indices, attr_map.shape[:2])
    return topk_coords

def drop_features(img_tensor, drop_coords, drop_value=0):
    """Discard the feature at the specified coordinates (set the area to be discarded as drop_ralue)"""
    img_dropped = img_tensor.clone()
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    drop_mask = np.zeros((H, W), dtype=bool)
    drop_mask[drop_coords] = True
    img_dropped[:, drop_mask] = drop_value
    return img_dropped

def visualize_attr_maps(img_original, attr_maps, method_names, save_path='attr_maps_all.png'):
    n_methods = len(method_names)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4*(n_methods + 1), 4))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, (attr_map, method) in enumerate(zip(attr_maps, method_names)):
        ax = axes[i+1]
        im = ax.imshow(attr_map, cmap='jet')
        ax.set_title(f'Saliency Map of {method}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_drop_results(img_original, dropped_imgs, method_names, drop_ratio, device, save_path='drop_results_all.png'):
    n_methods = len(method_names)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4*(n_methods + 1), 4))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, (img_dropped, method) in enumerate(zip(dropped_imgs, method_names)):
        ax = axes[i+1]
        ax.imshow(denormalize(img_dropped, device))
        ax.set_title(f'{method} Removing {drop_ratio*100}% Features')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_single_sample_result(img_idx, true_class, true_label, exp_result, drop_ratio, save_dir):
    result_path = os.path.join(save_dir, f'sample_{img_idx:04d}_result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(f"img_idx: {img_idx}\n")
        f.write(f"true_class: {true_class}\n")
        f.write(f"orig_class: {exp_result['orig_class']}\n")
        f.write(f"drop_ratio:{drop_ratio*100}%\n")
        f.write("="*50 + "\n")
        for method in exp_result['method_names']:
            res = exp_result['method_results'][method]
            f.write(f"{method}：\n")
            f.write(f"  dropped_class: {res['dropped_class']}\n")
            f.write(f"  consistency: {res['consistency']}\n")
            f.write("\n")

# -------------------------- Feature importance generation --------------------------
def get_attribute_maps(model, img_tensor, device):
    attr_maps = {}
    
    # -------------------------- LIME --------------------------
    img_pil = denormalize(img_tensor.squeeze(0), device)
    img_np = np.array(img_pil) / 255.0
    
    def lime_predict_fn(images):
        images_tensor = torch.tensor(images.transpose(0, 3, 1, 2)).float().to(device)
        images_tensor = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_tensor)
        with torch.no_grad():
            outputs = model(images_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    from skimage.segmentation import slic
    
    def custom_segmentation(image):
        segments = slic(image)
        return segments
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, lime_predict_fn, top_labels=1, hide_color=0, num_samples=1000, segmentation_fn=custom_segmentation
    )
    top_label = explanation.top_labels[0]
    segments = explanation.segments
    local_exp = explanation.local_exp[top_label]

    lime_attr_combined = np.zeros(segments.shape)

    for seg_idx, score in local_exp:
        mask = (segments == seg_idx)
        lime_attr_combined[mask] = abs(score)

    if lime_attr_combined.max() > lime_attr_combined.min():
        lime_attr_normalized = (lime_attr_combined - lime_attr_combined.min()) / (lime_attr_combined.max() - lime_attr_combined.min())
    else:
        lime_attr_normalized = lime_attr_combined

    attr_maps['LIME'] = lime_attr_normalized
    
    # -------------------------- SHAP --------------------------
    background = torch.zeros((10, 3, 32, 32)).to(device)
    background = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(background)
    explainer = shap.GradientExplainer(
        (model, model.features[-3]), background, local_smoothing=0.5
    )
    img_batch = img_tensor.unsqueeze(0).to(device)
    shap_values = explainer.shap_values(img_batch)
    top_label = model(img_batch).argmax(dim=1).item()

    # Processing SHAP values: Channel average → 4 × 4 → Upsampling to 32 × 32
    shap_attr_4x4  = np.mean(np.abs(shap_values[top_label][0]), axis=0)  # (4,4)
    # Convert to tensor and upsample (using bilinear interpolation to maintain feature smoothness)
    shap_attr_tensor = torch.tensor(shap_attr_4x4).unsqueeze(0).unsqueeze(0).float()  # (1,1,4,4)
    shap_attr_32x32  = torch.nn.functional.interpolate(
        shap_attr_tensor, 
        size=(32, 32),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0).numpy()

    # Normalization
    shap_attr_32x32 = (shap_attr_32x32 - shap_attr_32x32.min()) / (shap_attr_32x32.max() - shap_attr_32x32.min() + 1e-8)
    attr_maps['SHAP'] = shap_attr_32x32
    
    # -------------------------- CAM --------------------------
    # VGG16 uses the last convolutional layer as the target layer
    target_layer = model.features[-3]
    cam = GradCAM(model=model, target_layers=[target_layer])
    input_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        target_category = output.argmax(dim=1).item()

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(target_category)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    cam_attr = grayscale_cam[0]
    attr_maps['Grad-CAM'] = cam_attr
    
    # -------------------------- Integrated Gradients --------------------------
    ig = IntegratedGradients(model)
    input_tensor = img_tensor.unsqueeze(0).to(device)
    target_label = model(input_tensor).argmax(dim=1).item()
    baseline = torch.zeros_like(input_tensor).to(device)
    baseline = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(baseline)
    ig_attr, _ = ig.attribute(
        input_tensor, baseline, target=target_label, return_convergence_delta=True
    )
    ig_attr = ig_attr.squeeze(0).cpu().numpy()
    ig_attr = np.mean(np.abs(ig_attr), axis=0)

    ig_attr_normalized = (ig_attr - ig_attr.min()) / (ig_attr.max() - ig_attr.min() + 1e-8)
    attr_maps['IG'] = ig_attr_normalized
    
    return attr_maps

# -------------------------- Feature dropout and consistency calculation --------------------------
def drop_experiment_all_methods(model, img_tensor, device, drop_ratio=0.5):
    attr_maps = get_attribute_maps(model, img_tensor, device)
    method_names = list(attr_maps.keys())  # ['LIME', 'SHAP', 'Grad-CAM', 'IG']
    
    with torch.no_grad():
        orig_output = model(img_tensor.unsqueeze(0).to(device))
        orig_pred = orig_output.argmax(dim=1).item()
        orig_class = class_names[orig_pred]
        orig_confidence = torch.softmax(orig_output, dim=1)[0, orig_pred].item()
    
    results = {}
    dropped_imgs = []  # Store images discarded by various methods (for visualization)
    
    for method in method_names:
        drop_coords = get_topk_pixels(attr_maps[method], drop_ratio=drop_ratio)
        img_dropped = drop_features(img_tensor, drop_coords)
        dropped_imgs.append(img_dropped)
        with torch.no_grad():
            dropped_output = model(img_dropped.unsqueeze(0).to(device))
            dropped_pred = dropped_output.argmax(dim=1).item()
            dropped_class = class_names[dropped_pred]
            dropped_confidence = torch.softmax(dropped_output, dim=1)[0, dropped_pred].item()
        consistency = 1 if orig_pred == dropped_pred else 0

        results[method] = {
            'orig_class': orig_class,
            'dropped_class': dropped_class,
            'consistency': consistency,
            'attr_map': attr_maps[method],
            'orig_pred': orig_pred
        }
    
    return {
        'orig_class': orig_class,
        'orig_pred': orig_pred,
        'method_results': results,
        'dropped_imgs': dropped_imgs,
        'method_names': method_names,
        'attr_maps_list': [attr_maps[method] for method in method_names]
    }

if __name__ == "__main__":
    total_start = time.time()
    result_dir = 'vgg16/result_0.7'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # -------------------------- configuration and data preparation --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    DROP_RATIO = 0.7
    NUM_SAMPLES = 10000
    RANDOM_SEED = 42

    train_transform, test_transform = get_data_transforms()
    #train_dataset = torchvision.datasets.CIFAR10(
    #    root='./data', train=True, download=True, transform=train_transform
    #)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    #)
    #test_loader = torch.utils.data.DataLoader(
    #    test_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True
    #)
    
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # -------------------------- Initialize the model and train it --------------------------
    model = VGG16().to(device)
    
    # train
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("\n=== 开始训练VGG16 ===")
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "vgg16_cifar10_best.pth")
            print(f"保存最优权重（准确率: {best_accuracy:.2f}%）")
    print(f"\n训练完成！最优测试准确率: {best_accuracy:.2f}%")
    """

    # Load the optimal weights
    model.load_state_dict(torch.load("vgg16_cifar10_best.pth", map_location=device, weights_only=True))
    model.eval()
    
    batch_results = {
        'LIME': {'consistencies': []},
        'SHAP': {'consistencies': []},
        'Grad-CAM': {'consistencies': []},
        'IG': {'consistencies': []}
    }

    total_exp_samples = 0
    correct_exp_samples = 0  # The number of correct samples predicted by the model
    
    for img_idx, (img_tensor, true_label) in enumerate(tqdm(test_loader, desc="experimental progress")):
        img_tensor = img_tensor[0].to(device)
        true_label_single = true_label[0]
        true_class = class_names[true_label_single.item()]
        total_exp_samples += 1
        
        exp_result = drop_experiment_all_methods(model, img_tensor, device, drop_ratio=DROP_RATIO)
        method_results = exp_result['method_results']
        orig_pred = exp_result['orig_pred']

        if orig_pred == true_label_single:
            correct_exp_samples += 1
        
        for method in batch_results.keys():
            consistency = method_results[method]['consistency']
            batch_results[method]['consistencies'].append(consistency)

        save_single_sample_result(
            img_idx=img_idx,
            true_class=true_class,
            true_label=true_label_single,
            exp_result=exp_result,
            drop_ratio=DROP_RATIO,
            save_dir=result_dir
        )
        
        img_original = denormalize(img_tensor, device)
        visualize_attr_maps(
            img_original,
            exp_result['attr_maps_list'],
            exp_result['method_names'],
            save_path=f'{result_dir}/sample_{img_idx:04d}_attr_maps.png'
        )
        
        visualize_drop_results(
            img_original,
            exp_result['dropped_imgs'],
            exp_result['method_names'],
            DROP_RATIO,
            device,
            save_path=f'{result_dir}/sample_{img_idx:04d}_drop_results.png'
        )

    exp_accuracy = 100 * correct_exp_samples / total_exp_samples if total_exp_samples > 0 else 0.0
    print("\n=== Accuracy of the model on experimental samples ===")
    print(f"total_exp_samples: {total_exp_samples}")
    print(f"correct_exp_samples: {correct_exp_samples}")
    print(f"exp_accuracy: {exp_accuracy:.2f}%")
    
    avg_consistencies = {}
    method_names = list(batch_results.keys())
    
    for method in method_names:
        avg_consist = np.mean(batch_results[method]['consistencies'])
        avg_consistencies[method] = avg_consist
    
    print("\n=== Consistency score summary ===")
    print(f"Number of test samples: {NUM_SAMPLES}")
    print(f"Proportion of discarded features: {DROP_RATIO*100}%")
    print("-" * 50)
    for method in method_names:
        print(f"{method} Average consistency score: {avg_consistencies[method]:.4f}")
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    methods_display = ['LIME', 'SHAP', 'Grad-CAM', 'Integrated Gradients']
    
    bars = plt.bar(
        methods_display,
        [avg_consistencies[method] for method in method_names],
        color=colors,
        alpha=0.8
    )
    
    plt.ylim(0, 1.0)
    plt.ylabel('Consistency score', fontsize=12)
    plt.title(f'Comparison of Consistency among Different Interpreters(Removing{DROP_RATIO*100}% Features)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [avg_consistencies[method] for method in method_names]):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/all_methods_consistency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    total_end = time.time()
    total_duration = total_end - total_start

    summary_path = os.path.join(result_dir, 'samples_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("Summary results\n")
        f.write("="*50 + "\n")
        f.write(f"Total number of experimental samples: {total_exp_samples}\n")
        f.write(f"\n=== total_duration: {timedelta(seconds=total_duration)} ===\n")
        f.write(f"Model accuracy: {exp_accuracy:.2f}%\n")
        f.write(f"Feature dropout ratio: {DROP_RATIO*100}%\n")
        f.write("\nAverage consistency score of each method:\n")
        for method in method_names:
            f.write(f"{method}: {avg_consistencies[method]:.4f}\n")
    
    print(f"\nDone! All results have been saved to: {result_dir}")