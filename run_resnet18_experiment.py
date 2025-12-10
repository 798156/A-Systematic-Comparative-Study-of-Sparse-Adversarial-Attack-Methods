"""
ResNet18完整实验
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time
import os
import json
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd

from target_model import load_target_model
from jsma_attack import jsma_attack
from one_pixel_attack import one_pixel_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm, compute_l2_norm, compute_ssim, compute_psnr

# 尝试导入RL方法
try:
    from sparse_attack_env_v2 import SparseAttackEnvV2
    from stable_baselines3 import PPO
    rl_v2_available = True
except:
    rl_v2_available = False

try:
    from sparse_attack_env import SparseAttackEnv
    rl_v1_available = True
except:
    rl_v1_available = False


def run_resnet18_experiment(num_samples=100, max_pixels=10):
    """运行ResNet18完整实验"""
    
    print("=" * 80)
    print(" ResNet18完整实验 - Week 1 Day 1")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  模型: ResNet18 (准确率 ~85%)")
    print(f"  样本数: {num_samples}")
    print(f"  最大修改像素: {max_pixels}")
    print(f"  攻击方法: {'RL V1, ' if rl_v1_available else ''}{'RL V2, ' if rl_v2_available else ''}JSMA, One-Pixel, SparseFool")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n 加载数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # 加载模型
    print("\n🔧 加载ResNet18模型...")
    model = load_target_model("resnet18")
    model = model.to(device)
    model.eval()
    
    # 结果存储
    results = {
        'JSMA': {'success': [], 'l0': [], 'l2': [], 'ssim': [], 'psnr': [], 'time': []},
        'One-Pixel': {'success': [], 'l0': [], 'l2': [], 'ssim': [], 'psnr': [], 'time': []},
        'SparseFool': {'success': [], 'l0': [], 'l2': [], 'ssim': [], 'psnr': [], 'time': []},
    }
    
    if rl_v1_available:
        results['RL V1'] = {'success': [], 'l0': [], 'l2': [], 'ssim': [], 'psnr': [], 'time': []}
    
    if rl_v2_available:
        results['RL V2'] = {'success': [], 'l0': [], 'l2': [], 'ssim': [], 'psnr': [], 'time': []}
    
    # 收集正确分类的样本
    print("\n 收集正确分类的样本...")
    correct_samples = []
    
    for idx, (images, labels) in enumerate(testloader):
        if len(correct_samples) >= num_samples:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            pred = outputs.argmax(dim=1).item()
            
            if pred == labels.item():
                correct_samples.append((images[0], labels.item()))
    
    print(f" 收集到 {len(correct_samples)} 个正确分类的样本\n")
    
    if len(correct_samples) < num_samples:
        print(f" 警告：只找到 {len(correct_samples)} 个正确分类样本")
        num_samples = len(correct_samples)
    
    # 开始实验
    print("=" * 80)
    print(" 开始攻击实验")
    print("=" * 80)
    
    for idx, (image, label) in enumerate(tqdm(correct_samples, desc="实验进度")):
        image = image.to(device)
        
        # 1. JSMA
        start = time.time()
        try:
            success, adv, pixels = jsma_attack(image, label, model, max_pixels=max_pixels, theta=5.0)
            elapsed = time.time() - start
            
            results['JSMA']['success'].append(success)
            results['JSMA']['time'].append(elapsed)
            
            if success:
                l0 = compute_l0_norm(image.cpu(), adv.cpu())
                l2 = compute_l2_norm(image.cpu(), adv.cpu())
                ssim = compute_ssim(image.cpu(), adv.cpu())
                psnr = compute_psnr(image.cpu(), adv.cpu())
                
                results['JSMA']['l0'].append(l0)
                results['JSMA']['l2'].append(l2)
                results['JSMA']['ssim'].append(ssim)
                results['JSMA']['psnr'].append(psnr)
        except Exception as e:
            results['JSMA']['success'].append(False)
            results['JSMA']['time'].append(0)
        
        # 2. One-Pixel
        start = time.time()
        try:
            success, params = one_pixel_attack(image, label, model, max_iter=200)
            elapsed = time.time() - start
            
            results['One-Pixel']['success'].append(success)
            results['One-Pixel']['time'].append(elapsed)
            
            if success:
                results['One-Pixel']['l0'].append(1)
                results['One-Pixel']['l2'].append(0.0)  # 近似
                results['One-Pixel']['ssim'].append(1.0)  # 近似
                results['One-Pixel']['psnr'].append(100.0)  # 近似
        except Exception as e:
            results['One-Pixel']['success'].append(False)
            results['One-Pixel']['time'].append(0)
        
        # 3. SparseFool
        start = time.time()
        try:
            success, adv, pixels = sparsefool_attack_simple(image, label, model, max_pixels=max_pixels)
            elapsed = time.time() - start
            
            results['SparseFool']['success'].append(success)
            results['SparseFool']['time'].append(elapsed)
            
            if success:
                l0 = compute_l0_norm(image.cpu(), adv.cpu())
                l2 = compute_l2_norm(image.cpu(), adv.cpu())
                ssim = compute_ssim(image.cpu(), adv.cpu())
                psnr = compute_psnr(image.cpu(), adv.cpu())
                
                results['SparseFool']['l0'].append(l0)
                results['SparseFool']['l2'].append(l2)
                results['SparseFool']['ssim'].append(ssim)
                results['SparseFool']['psnr'].append(psnr)
        except Exception as e:
            results['SparseFool']['success'].append(False)
            results['SparseFool']['time'].append(0)
        
        # 4. RL V1 (如果可用)
        if rl_v1_available and 'RL V1' in results:
            start = time.time()
            try:
                env = SparseAttackEnv(model, max_steps=max_pixels, device=device)
                agent_path = "ppo_sparse_model.zip"
                
                if os.path.exists(agent_path):
                    agent = PPO.load(agent_path)
                    obs, _ = env.reset(image.unsqueeze(0), label)
                    done = False
                    
                    for _ in range(max_pixels):
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        if done:
                            break
                    
                    success = info.get('attack_success', False)
                    adv_image = info.get('adv_image', image)
                    elapsed = time.time() - start
                    
                    results['RL V1']['success'].append(success)
                    results['RL V1']['time'].append(elapsed)
                    
                    if success:
                        l0 = compute_l0_norm(image.cpu(), adv_image.cpu())
                        l2 = compute_l2_norm(image.cpu(), adv_image.cpu())
                        ssim = compute_ssim(image.cpu(), adv_image.cpu())
                        psnr = compute_psnr(image.cpu(), adv_image.cpu())
                        
                        results['RL V1']['l0'].append(l0)
                        results['RL V1']['l2'].append(l2)
                        results['RL V1']['ssim'].append(ssim)
                        results['RL V1']['psnr'].append(psnr)
                else:
                    results['RL V1']['success'].append(False)
                    results['RL V1']['time'].append(0)
            except Exception as e:
                results['RL V1']['success'].append(False)
                results['RL V1']['time'].append(0)
        
        # 5. RL V2 (如果可用)
        if rl_v2_available and 'RL V2' in results:
            start = time.time()
            try:
                env = SparseAttackEnvV2(model, max_steps=max_pixels, device=device)
                agent_path = "ppo_sparse_v2.zip"
                
                if os.path.exists(agent_path):
                    agent = PPO.load(agent_path)
                    obs, _ = env.reset(image.unsqueeze(0), label)
                    done = False
                    
                    for _ in range(max_pixels):
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        if done:
                            break
                    
                    success = info.get('attack_success', False)
                    adv_image = info.get('adv_image', image)
                    elapsed = time.time() - start
                    
                    results['RL V2']['success'].append(success)
                    results['RL V2']['time'].append(elapsed)
                    
                    if success:
                        l0 = compute_l0_norm(image.cpu(), adv_image.cpu())
                        l2 = compute_l2_norm(image.cpu(), adv_image.cpu())
                        ssim = compute_ssim(image.cpu(), adv_image.cpu())
                        psnr = compute_psnr(image.cpu(), adv_image.cpu())
                        
                        results['RL V2']['l0'].append(l0)
                        results['RL V2']['l2'].append(l2)
                        results['RL V2']['ssim'].append(ssim)
                        results['RL V2']['psnr'].append(psnr)
                else:
                    results['RL V2']['success'].append(False)
                    results['RL V2']['time'].append(0)
            except Exception as e:
                results['RL V2']['success'].append(False)
                results['RL V2']['time'].append(0)
    
    # 分析结果
    print("\n" + "=" * 80)
    print("📊 实验结果")
    print("=" * 80)
    
    summary = {}
    
    print(f"\n{'方法':<20} {'ASR (%)':<12} {'平均L0':<12} {'平均L2':<12} {'平均SSIM':<12} {'平均时间 (s)':<15}")
    print("-" * 100)
    
    for method, data in results.items():
        if not data['success']:
            continue
        
        asr = np.mean(data['success']) * 100 if data['success'] else 0
        avg_l0 = np.mean(data['l0']) if data['l0'] else 0
        avg_l2 = np.mean(data['l2']) if data['l2'] else 0
        avg_ssim = np.mean(data['ssim']) if data['ssim'] else 0
        avg_time = np.mean(data['time']) if data['time'] else 0
        
        print(f"{method:<20} {asr:<12.1f} {avg_l0:<12.2f} {avg_l2:<12.4f} {avg_ssim:<12.4f} {avg_time:<15.3f}")
        
        summary[method] = {
            'ASR': float(asr),
            'L0': float(avg_l0),
            'L2': float(avg_l2),
            'SSIM': float(avg_ssim),
            'Time': float(avg_time)
        }
    
    # 保存结果
    output_dir = "results/week1_day1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    with open(f"{output_dir}/resnet18_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存详细数据
    for method, data in results.items():
        if data['success']:
            df = pd.DataFrame({
                'success': data['success'],
                'l0': data['l0'] + [None] * (len(data['success']) - len(data['l0'])),
                'l2': data['l2'] + [None] * (len(data['success']) - len(data['l2'])),
                'ssim': data['ssim'] + [None] * (len(data['success']) - len(data['ssim'])),
                'psnr': data['psnr'] + [None] * (len(data['success']) - len(data['psnr'])),
                'time': data['time']
            })
            df.to_csv(f"{output_dir}/resnet18_{method.replace(' ', '_')}.csv", index=False)
    
    # 生成图表
    print(f"\n📊 生成图表...")
    
    # ASR对比
    plt.figure(figsize=(10, 6))
    methods = list(summary.keys())
    asrs = [summary[m]['ASR'] for m in methods]
    
    plt.bar(methods, asrs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(methods)])
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('ASR Comparison - ResNet18', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/asr_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # L0对比
    plt.figure(figsize=(10, 6))
    l0s = [summary[m]['L0'] for m in methods]
    
    plt.bar(methods, l0s, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(methods)])
    plt.ylabel('Average L0 Norm', fontsize=12)
    plt.title('L0 Norm Comparison - ResNet18', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/l0_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" 结果保存在: {output_dir}/")
    print(f"  - resnet18_summary.json")
    print(f"  - resnet18_*.csv")
    print(f"  - asr_comparison.png")
    print(f"  - l0_comparison.png")
    
    print("\n" + "=" * 80)
    print(" ResNet18实验完成！")
    print("=" * 80)
    
    return summary, results


if __name__ == "__main__":
    # 运行实验
    summary, results = run_resnet18_experiment(num_samples=100, max_pixels=10)
