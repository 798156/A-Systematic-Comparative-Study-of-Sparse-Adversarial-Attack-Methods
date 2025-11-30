#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失败案例分析脚本
分析哪些样本难以攻击，为什么失败
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FailureCaseAnalyzer:
    def __init__(self):
        self.results_dir = Path('results/complete_baseline')
        self.output_dir = Path('results/failure_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = ['resnet18', 'vgg16', 'mobilenetv2']
        self.methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        # 存储数据
        self.all_data = {}
        self.failure_samples = []
        self.partial_failure_samples = []
        self.hard_samples = []  # 所有方法都失败的样本
        
    def load_all_data(self):
        """加载所有实验数据"""
        print("\n" + "="*60)
        print(" 加载实验数据...")
        print("="*60)
        
        for model in self.models:
            self.all_data[model] = {}
            for method in self.methods:
                json_file = self.results_dir / f'{model}_{method}.json'
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        self.all_data[model][method] = data
                        print(f"✓ 加载: {model}_{method}")
                else:
                    print(f"  文件不存在: {json_file}")
        
        print(f"\n 加载完成: {len(self.models)}个模型 × {len(self.methods)}个方法")
    
    def identify_failure_samples(self):
        """识别失败样本"""
        print("\n" + "="*60)
        print("识别失败样本...")
        print("="*60)
        
        # 使用ResNet18作为代表模型进行分析
        model = 'resnet18'
        model_data = self.all_data[model]
        
        # 获取样本数量
        first_method = list(model_data.values())[0]
        if 'detailed_results' in first_method:
            num_samples = len(first_method['detailed_results'])
        else:
            num_samples = 100  # 默认
        
        print(f"分析模型: {model}")
        print(f"样本数量: {num_samples}")
        
        # 分析每个样本
        for sample_id in range(num_samples):
            successes = []
            failures = []
            
            for method in self.methods:
                if method in model_data:
                    results = model_data[method].get('detailed_results', [])
                    if sample_id < len(results):
                        sample_result = results[sample_id]
                        if sample_result.get('success', False):
                            successes.append(method)
                        else:
                            failures.append(method)
            
            # 分类样本
            if len(failures) == len(self.methods):
                # 所有方法都失败 = 硬样本
                self.hard_samples.append({
                    'sample_id': sample_id,
                    'model': model,
                    'failed_methods': failures,
                    'success_count': 0
                })
            elif len(failures) > 0:
                # 部分方法失败
                self.partial_failure_samples.append({
                    'sample_id': sample_id,
                    'model': model,
                    'failed_methods': failures,
                    'success_methods': successes,
                    'success_count': len(successes)
                })
        
        # 统计
        print(f"\ 失败样本统计:")
        print(f"  • 硬样本（所有方法都失败）: {len(self.hard_samples)} ({len(self.hard_samples)/num_samples*100:.1f}%)")
        print(f"  • 部分失败样本: {len(self.partial_failure_samples)} ({len(self.partial_failure_samples)/num_samples*100:.1f}%)")
        print(f"  • 易攻击样本（所有方法都成功）: {num_samples - len(self.hard_samples) - len(self.partial_failure_samples)}")
    
    def analyze_hard_samples(self):
        """分析硬样本特征"""
        print("\n" + "="*60)
        print("🔬 分析硬样本...")
        print("="*60)
        
        if len(self.hard_samples) == 0:
            print("  没有找到硬样本（所有样本至少被一种方法成功攻击）")
            return None
        
        print(f"\n找到 {len(self.hard_samples)} 个硬样本")
        print("这些样本对所有5种攻击方法都免疫！")
        
        # 可以进一步分析（如果有标签信息）
        # 这里只生成基本统计
        
        return {
            'num_hard_samples': len(self.hard_samples),
            'percentage': len(self.hard_samples) / 100 * 100,
            'sample_ids': [s['sample_id'] for s in self.hard_samples]
        }
    
    def analyze_method_specific_failures(self):
        """分析方法特定的失败模式"""
        print("\n" + "="*60)
        print("分析方法特定失败模式...")
        print("="*60)
        
        # 统计每个方法的失败次数
        method_failures = Counter()
        for sample in self.partial_failure_samples:
            for method in sample['failed_methods']:
                method_failures[method] += 1
        
        # 统计哪些方法组合容易失败
        failure_patterns = Counter()
        for sample in self.partial_failure_samples:
            pattern = tuple(sorted(sample['failed_methods']))
            failure_patterns[pattern] += 1
        
        print(f"\n方法失败频率:")
        for method, count in method_failures.most_common():
            percentage = count / len(self.partial_failure_samples) * 100
            print(f"  • {method:15s}: {count:3d} 次 ({percentage:.1f}%)")
        
        print(f"\n最常见的失败组合（Top 5）:")
        for i, (pattern, count) in enumerate(failure_patterns.most_common(5), 1):
            methods_str = ', '.join(pattern)
            print(f"  {i}. [{methods_str}]: {count} 次")
        
        return {
            'method_failures': dict(method_failures),
            'failure_patterns': dict(failure_patterns)
        }
    
    def analyze_success_patterns(self):
        """分析成功模式"""
        print("\n" + "="*60)
        print(" 分析成功模式...")
        print("="*60)
        
        # 在部分失败样本中，哪些方法最可靠？
        method_successes = Counter()
        for sample in self.partial_failure_samples:
            for method in sample['success_methods']:
                method_successes[method] += 1
        
        total_partial = len(self.partial_failure_samples)
        
        print(f"\n在{total_partial}个部分失败样本中，各方法的成功率:")
        for method, count in method_successes.most_common():
            percentage = count / total_partial * 100
            print(f"  • {method:15s}: {count:3d}/{total_partial} = {percentage:.1f}%")
        
        return dict(method_successes)
    
    def generate_visualizations(self):
        """生成可视化"""
        print("\n" + "="*60)
        print(" 生成可视化...")
        print("="*60)
        
        # 1. 失败样本分布
        self._plot_failure_distribution()
        
        # 2. 方法失败率对比
        self._plot_method_failure_rates()
        
        # 3. 成功率vs难度
        self._plot_difficulty_distribution()
        
        print("所有可视化已生成")
    
    def _plot_failure_distribution(self):
        """绘制失败样本分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：样本分类
        categories = ['All Failed\n(Hard)', 'Partial Failed', 'All Success\n(Easy)']
        counts = [
            len(self.hard_samples),
            len(self.partial_failure_samples),
            100 - len(self.hard_samples) - len(self.partial_failure_samples)
        ]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        
        ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Samples', fontweight='bold')
        ax1.set_title('Sample Classification by Attack Difficulty', 
                     fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加百分比标签
        for i, (cat, count) in enumerate(zip(categories, counts)):
            percentage = count / 100 * 100
            ax1.text(i, count + 1, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 右图：成功方法数量分布
        success_counts = Counter()
        for sample in self.partial_failure_samples:
            success_counts[sample['success_count']] += 1
        
        # 包括完全失败和完全成功
        success_counts[0] = len(self.hard_samples)
        success_counts[5] = 100 - len(self.hard_samples) - len(self.partial_failure_samples)
        
        x = sorted(success_counts.keys())
        y = [success_counts[i] for i in x]
        
        ax2.bar(x, y, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Successful Methods', fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontweight='bold')
        ax2.set_title('Distribution of Success Count per Sample',
                     fontweight='bold', fontsize=12)
        ax2.set_xticks(range(6))
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'failure_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'failure_distribution.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ failure_distribution.pdf")
    
    def _plot_method_failure_rates(self):
        """绘制方法失败率"""
        # 统计每个方法的总失败率
        method_stats = {}
        model = 'resnet18'
        
        for method in self.methods:
            data = self.all_data[model][method]
            total = data.get('total_samples', 100)
            success = data.get('success_count', 0)
            failure = total - success
            failure_rate = failure / total * 100
            
            method_stats[method] = {
                'failure_rate': failure_rate,
                'failure_count': failure,
                'total': total
            }
        
        # 排序
        sorted_methods = sorted(method_stats.items(), 
                               key=lambda x: x[1]['failure_rate'], 
                               reverse=True)
        
        methods_display = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = [methods_display[m] for m, _ in sorted_methods]
        rates = [stats['failure_rate'] for _, stats in sorted_methods]
        counts = [stats['failure_count'] for _, stats in sorted_methods]
        
        bars = ax.barh(labels, rates, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Failure Rate (%)', fontweight='bold')
        ax.set_title('Method-Specific Failure Rates (ResNet18)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (rate, count) in enumerate(zip(rates, counts)):
            ax.text(rate + 1, i, f'{rate:.1f}% ({count})', 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_failure_rates.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'method_failure_rates.pdf', bbox_inches='tight')
        plt.close()
        print("  ✓ method_failure_rates.pdf")
    
    def _plot_difficulty_distribution(self):
        """绘制难度分布"""
        # 按成功方法数分组
        difficulty_groups = defaultdict(list)
        
        # 硬样本
        for _ in self.hard_samples:
            difficulty_groups[0].append(0)
        
        # 部分失败
        for sample in self.partial_failure_samples:
            difficulty_groups[sample['success_count']].append(sample['success_count'])
        
        # 完全成功
        easy_count = 100 - len(self.hard_samples) - len(self.partial_failure_samples)
        for _ in range(easy_count):
            difficulty_groups[5].append(5)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_labels = ['0\n(Hardest)', '1', '2', '3', '4', '5\n(Easiest)']
        x_pos = range(6)
        heights = [len(difficulty_groups[i]) for i in range(6)]
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 6))
        
        bars = ax.bar(x_pos, heights, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Number of Successful Attacks (Difficulty Level)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax.set_title('Sample Distribution by Attack Difficulty',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加标签
        for i, (bar, h) in enumerate(zip(bars, heights)):
            percentage = h / 100 * 100
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                   f'{h}\n({percentage:.0f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'difficulty_distribution.pdf', bbox_inches='tight')
        plt.close()
        print("  difficulty_distribution.pdf")
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print(" 生成分析报告...")
        print("="*60)
        
        report = f"""# 失败案例分析报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析模型:** ResNet18  
**总样本数:** 100

---

## 执行摘要

本报告分析了5种L0攻击方法在ResNet18模型上的失败案例，
旨在识别"硬样本"（难以攻击的样本）并理解失败原因。

---

## 1. 样本分类统计

### 1.1 整体分布

| 类别 | 数量 | 百分比 | 定义 |
|------|------|--------|------|
| **硬样本** | {len(self.hard_samples)} | {len(self.hard_samples)}% | 所有5种方法都失败 |
| **部分失败样本** | {len(self.partial_failure_samples)} | {len(self.partial_failure_samples)}% | 1-4种方法失败 |
| **易攻击样本** | {100 - len(self.hard_samples) - len(self.partial_failure_samples)} | {100 - len(self.hard_samples) - len(self.partial_failure_samples)}% | 所有5种方法都成功 |

### 1.2 关键发现

"""
        
        if len(self.hard_samples) == 0:
            report += " **没有发现硬样本**！所有100个样本都至少被一种方法成功攻击。\n"
            report += "这表明：当前的5种攻击方法已经能够覆盖ResNet18的大部分决策空间。\n\n"
        else:
            report += f" **发现{len(self.hard_samples)}个硬样本**！这些样本对所有攻击方法都免疫。\n"
            report += f"占比：{len(self.hard_samples)}%\n"
            report += f"样本ID：{[s['sample_id'] for s in self.hard_samples[:10]]}\n\n"
        
        report += "---\n\n## 2. 方法特定失败分析\n\n"
        report += "### 2.1 方法失败率\n\n"
        
        # 方法失败率
        model = 'resnet18'
        report += "| 方法 | 总失败数 | 失败率 | 排名 |\n"
        report += "|------|----------|--------|------|\n"
        
        method_failures = []
        for method in self.methods:
            data = self.all_data[model][method]
            total = data.get('total_samples', 100)
            success = data.get('success_count', 0)
            failure = total - success
            failure_rate = failure / total * 100
            method_failures.append((method, failure, failure_rate))
        
        method_failures.sort(key=lambda x: x[2], reverse=True)
        
        methods_display = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        for rank, (method, count, rate) in enumerate(method_failures, 1):
            report += f"| {methods_display[method]} | {count} | {rate:.1f}% | #{rank} |\n"
        
        report += "\n### 2.2 分析\n\n"
        
        # 找出最脆弱和最鲁棒的方法
        weakest = method_failures[0]
        strongest = method_failures[-1]
        
        report += f"- **最脆弱方法:** {methods_display[weakest[0]]} ({weakest[2]:.1f}%失败率)\n"
        report += f"- **最鲁棒方法:** {methods_display[strongest[0]]} ({strongest[2]:.1f}%失败率)\n"
        report += f"- **失败率差异:** {weakest[2] - strongest[2]:.1f}个百分点\n\n"
        
        report += "---\n\n## 3. 难度级别分析\n\n"
        report += "### 3.1 按成功方法数分组\n\n"
        
        difficulty_dist = Counter()
        difficulty_dist[0] = len(self.hard_samples)
        for sample in self.partial_failure_samples:
            difficulty_dist[sample['success_count']] += 1
        difficulty_dist[5] = 100 - len(self.hard_samples) - len(self.partial_failure_samples)
        
        report += "| 成功方法数 | 样本数 | 百分比 | 难度级别 |\n"
        report += "|-----------|--------|--------|----------|\n"
        for i in range(6):
            count = difficulty_dist[i]
            percentage = count / 100 * 100
            if i == 0:
                level = "极难"
            elif i <= 2:
                level = "困难"
            elif i <= 3:
                level = "中等"
            else:
                level = "容易"
            report += f"| {i} | {count} | {percentage:.0f}% | {level} |\n"
        
        report += "\n### 3.2 难度分布洞察\n\n"
        
        hard_medium = difficulty_dist[0] + difficulty_dist[1] + difficulty_dist[2]
        report += f"- **困难样本（0-2方法成功）:** {hard_medium}个 ({hard_medium}%)\n"
        easy = difficulty_dist[4] + difficulty_dist[5]
        report += f"- **容易样本（4-5方法成功）:** {easy}个 ({easy}%)\n"
        report += f"- **中等样本（3方法成功）:** {difficulty_dist[3]}个 ({difficulty_dist[3]}%)\n\n"
        
        report += "---\n\n## 4. 失败原因分析（推测）\n\n"
        report += "基于实验结果，我们推测失败的主要原因包括：\n\n"
        report += "### 4.1 梯度相关问题\n"
        report += "- **梯度消失/饱和:** 某些样本可能位于决策边界的平坦区域\n"
        report += "- **梯度爆炸:** 梯度过大导致修改过激\n"
        report += "- **梯度方向误导:** 梯度指向的方向不是最优攻击路径\n\n"
        
        report += "### 4.2 决策边界问题\n"
        report += "- **决策边界距离:** 某些样本距离决策边界很远\n"
        report += "- **边界复杂度:** 决策边界过于复杂，难以找到稀疏路径\n"
        report += "- **局部最优陷阱:** 贪心策略陷入局部最优\n\n"
        
        report += "### 4.3 样本特征\n"
        report += "- **高置信度预测:** 模型对这些样本的预测置信度很高\n"
        report += "- **类别语义:** 某些类别本质上更难攻击\n"
        report += "- **视觉特征:** 纹理简单或颜色单一的样本可能更难修改\n\n"
        
        report += "---\n\n## 5. 方法互补性分析\n\n"
        
        # 统计在部分失败样本中，各方法的表现
        if len(self.partial_failure_samples) > 0:
            method_successes = Counter()
            for sample in self.partial_failure_samples:
                for method in sample['success_methods']:
                    method_successes[method] += 1
            
            report += f"在{len(self.partial_failure_samples)}个部分失败样本中：\n\n"
            report += "| 方法 | 成功次数 | 成功率 | 互补价值 |\n"
            report += "|------|----------|--------|----------|\n"
            
            for method, count in method_successes.most_common():
                rate = count / len(self.partial_failure_samples) * 100
                if rate > 60:
                    value = "高"
                elif rate > 30:
                    value = "中"
                else:
                    value = "低"
                report += f"| {methods_display[method]} | {count} | {rate:.1f}% | {value} |\n"
        report_file = self.output_dir / 'failure_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 分析报告已保存: {report_file}")
        
        return report
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("\n" + ""*30)
        print("失败案例分析")
        
        # 1. 加载数据
        self.load_all_data()
        
        # 2. 识别失败样本
        self.identify_failure_samples()
        
        # 3. 分析硬样本
        hard_stats = self.analyze_hard_samples()
        
        # 4. 分析方法特定失败
        method_stats = self.analyze_method_specific_failures()
     
        success_stats = self.analyze_success_patterns()
     
        self.generate_visualizations()
      
        report = self.generate_report()
        
        # 最终总结
        print("\n" + ""*30)
        print("失败案例分析完成！")
        
        print(f"\n 生成的文件:")
        print(f"  1. {self.output_dir / 'failure_analysis_report.md'}")
        print(f"  2. {self.output_dir / 'failure_distribution.pdf'}")
        print(f"  3. {self.output_dir / 'method_failure_rates.pdf'}")
        print(f"  4. {self.output_dir / 'difficulty_distribution.pdf'}")
        
        print(f"\n 关键发现:")
        print(f"  • 硬样本数量: {len(self.hard_samples)}")
        print(f"  • 部分失败样本: {len(self.partial_failure_samples)}")
        print(f"  • 易攻击样本: {100 - len(self.hard_samples) - len(self.partial_failure_samples)}")
        
        if len(self.hard_samples) == 0:
            print(f"\n **重要发现**: 没有硬样本！所有样本都至少被一种方法成功攻击！")
            print(f"   这表明5种方法的组合已经能够覆盖大部分攻击场景。")

def main():
    analyzer = FailureCaseAnalyzer()
    analyzer.run_complete_analysis()
    return 0

if __name__ == '__main__':
    exit(main())















