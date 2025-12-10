#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询效率分析
分析不同攻击方法的时间开销、查询次数和效率权衡
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class QueryEfficiencyAnalyzer:
    def __init__(self):
        self.results_dir = Path('results/complete_baseline')
        self.output_dir = Path('results/query_efficiency')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = ['resnet18', 'vgg16', 'mobilenetv2']
        self.methods = ['jsma', 'sparsefool', 'greedy', 'pixelgrad', 'randomsparse']
        
        self.method_names = {
            'jsma': 'JSMA',
            'sparsefool': 'SparseFool',
            'greedy': 'Greedy',
            'pixelgrad': 'PixelGrad',
            'randomsparse': 'RandomSparse'
        }
        
        # 估算的查询次数（基于方法特性）
        self.query_estimates = {
            'jsma': {
                'queries_per_pixel': 2,  # 前向+显著性计算
                'description': '每次迭代需要计算显著性图'
            },
            'sparsefool': {
                'queries_per_iter': 1,  # 每次迭代一次前向
                'description': '迭代优化，每次迭代一次查询'
            },
            'greedy': {
                'queries_per_pixel': 1,  # 每个像素一次前向
                'description': '贪心选择，每个像素一次查询'
            },
            'pixelgrad': {
                'queries_per_pixel': 1,  # 梯度计算
                'description': '梯度引导，需要反向传播'
            },
            'randomsparse': {
                'queries_per_attempt': 1,  # 每次尝试一次前向
                'description': '随机采样，每次尝试一次查询'
            }
        }
        
        self.all_data = {}
    
    def load_all_data(self):
        """加载所有实验数据"""
        print("\n" + "="*60)
        print("加载实验数据...")
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
        
        print("加载完成")
    
    def analyze_time_statistics(self):
        """分析时间统计"""
        print("\n" + "="*60)
        print("分析时间统计...")
        print("="*60)
        
        time_stats = {}
        
        for model in self.models:
            time_stats[model] = {}
            for method in self.methods:
                if method not in self.all_data[model]:
                    continue
                
                data = self.all_data[model][method]
                
                # 提取时间数据
                times = []
                for result in data.get('detailed_results', []):
                    if 'time' in result:
                        times.append(result['time'])
                
                if times:
                    time_stats[model][method] = {
                        'avg_time': np.mean(times),
                        'std_time': np.std(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'median_time': np.median(times),
                        'total_samples': len(times)
                    }
        
        return time_stats
    
    def calculate_query_estimates(self, time_stats):
        """估算查询次数"""
        print("\n" + "="*60)
        print("估算查询次数...")
        print("="*60)
        
        query_stats = {}
        
        for model in self.models:
            query_stats[model] = {}
            for method in self.methods:
                if method not in self.all_data[model]:
                    continue
                
                data = self.all_data[model][method]
                avg_l0 = data.get('avg_l0', 0)
                
                # 基于方法类型估算查询次数
                if method == 'jsma':
                    # JSMA: 每个像素需要计算显著性
                    estimated_queries = int(avg_l0 * self.query_estimates['jsma']['queries_per_pixel'])
                elif method == 'sparsefool':
                    # SparseFool: 基于迭代次数（假设20次迭代）
                    estimated_queries = 20
                elif method == 'greedy':
                    # Greedy: 每个候选像素一次查询
                    estimated_queries = int(avg_l0 * 10)  # 假设每次从10个候选中选择
                elif method == 'pixelgrad':
                    # PixelGrad: 类似Greedy但有动量
                    estimated_queries = int(avg_l0 * 5)
                elif method == 'randomsparse':
                    # RandomSparse: 随机尝试次数（配置中是50）
                    estimated_queries = 50
                else:
                    estimated_queries = 0
                
                query_stats[model][method] = {
                    'estimated_queries': estimated_queries,
                    'avg_l0': avg_l0,
                    'avg_time': time_stats[model][method]['avg_time'] if method in time_stats[model] else 0
                }
        
        return query_stats
    
    def analyze_efficiency_tradeoff(self, time_stats):
        """分析效率权衡"""
        print("\n" + "="*60)
        print("分析效率权衡...")
        print("="*60)
        
        # 使用ResNet18作为主要分析对象
        model = 'resnet18'
        
        tradeoff_data = []
        
        for method in self.methods:
            if method not in self.all_data[model]:
                continue
            
            data = self.all_data[model][method]
            
            tradeoff_data.append({
                'method': self.method_names[method],
                'asr': data.get('asr', 0),
                'avg_time': time_stats[model][method]['avg_time'] if method in time_stats[model] else 0,
                'avg_l0': data.get('avg_l0', 0)
            })
        
        return tradeoff_data
    
    def generate_visualizations(self, time_stats, tradeoff_data):
        """生成可视化"""
        print("\n" + "="*60)
        print("生成可视化...")
        print("="*60)
        
        # 1. 时间对比柱状图
        self._plot_time_comparison(time_stats)
        
        # 2. 效率权衡散点图 (ASR vs Time)
        self._plot_efficiency_tradeoff(tradeoff_data)
        
        # 3. 时间-L0权衡
        self._plot_time_l0_tradeoff(tradeoff_data)
        
        print("✓ 可视化生成完成")
    
    def _plot_time_comparison(self, time_stats):
        """绘制时间对比"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for ax, model in zip(axes, self.models):
            methods_list = []
            times = []
            stds = []
            
            for method in self.methods:
                if method in time_stats[model]:
                    methods_list.append(self.method_names[method])
                    times.append(time_stats[model][method]['avg_time'])
                    stds.append(time_stats[model][method]['std_time'])
            
            x = np.arange(len(methods_list))
            bars = ax.bar(x, times, yerr=stds, capsize=5, alpha=0.8, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731', '#A29BFE'])
            
            ax.set_xlabel('Attack Method', fontweight='bold')
            ax.set_ylabel('Average Time (seconds)', fontweight='bold')
            ax.set_title(f'{model.upper()} - Attack Time Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods_list, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'time_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ time_comparison.pdf")
    
    def _plot_efficiency_tradeoff(self, tradeoff_data):
        """绘制效率权衡散点图"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        methods = [d['method'] for d in tradeoff_data]
        asrs = [d['asr'] for d in tradeoff_data]
        times = [d['avg_time'] for d in tradeoff_data]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731', '#A29BFE']
        
        scatter = ax.scatter(times, asrs, s=300, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        # 添加方法标签
        for i, method in enumerate(methods):
            ax.annotate(method, (times[i], asrs[i]), 
                       fontsize=11, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Average Time per Sample (seconds)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Attack Efficiency Trade-off: ASR vs Time', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        # 添加理想区域标注（右上角）
        ax.text(0.95, 0.95, 'Ideal Zone:\nHigh ASR\nLow Time', 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
               verticalalignment='top', horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'efficiency_tradeoff.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ efficiency_tradeoff.pdf")
    
    def _plot_time_l0_tradeoff(self, tradeoff_data):
        """绘制时间-L0权衡"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        methods = [d['method'] for d in tradeoff_data]
        l0s = [d['avg_l0'] for d in tradeoff_data]
        times = [d['avg_time'] for d in tradeoff_data]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731', '#A29BFE']
        
        scatter = ax.scatter(l0s, times, s=300, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        # 添加方法标签
        for i, method in enumerate(methods):
            ax.annotate(method, (l0s[i], times[i]), 
                       fontsize=11, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Average L0 Norm (pixels modified)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_title('Sparsity vs Computational Cost', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_l0_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'time_l0_tradeoff.pdf', bbox_inches='tight')
        plt.close()
        
        print("  ✓ time_l0_tradeoff.pdf")
    
    def generate_report(self, time_stats, query_stats, tradeoff_data):
        """生成分析报告"""
        print("\n" + "="*60)
        print("生成分析报告...")
        print("="*60)
        
        report = f"""# 查询效率分析报告

**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析对象:** ResNet18, VGG16, MobileNetV2  
**攻击方法:** 5种

---

## 1. 时间效率统计（ResNet18）

| 方法 | 平均时间 | 标准差 | 最小值 | 最大值 | 中位数 |
|------|---------|--------|--------|--------|--------|
"""
        
        model = 'resnet18'
        for method in self.methods:
            if method in time_stats[model]:
                stats = time_stats[model][method]
                name = self.method_names[method]
                report += f"| {name} | {stats['avg_time']:.3f}s | {stats['std_time']:.3f}s | {stats['min_time']:.3f}s | {stats['max_time']:.3f}s | {stats['median_time']:.3f}s |\n"
        
        report += "\n### 1.1 时间效率排名\n\n"
        
        sorted_methods = sorted(
            [(m, time_stats[model][m]['avg_time']) for m in self.methods if m in time_stats[model]],
            key=lambda x: x[1]
        )
        
        for rank, (method, time) in enumerate(sorted_methods, 1):
            name = self.method_names[method]
            if rank == 1
                report += f"{rank}. **{name}:** {time:.3f}s 最快\n"
            elif rank == len(sorted_methods):
                report += f"{rank}. **{name}:** {time:.3f}s 最慢\n"
            else:
                report += f"{rank}. **{name}:** {time:.3f}s\n"
        
        report += "\n---\n\n## 2. 查询次数估算\n\n"
        report += "基于方法特性估算的模型查询次数：\n\n"
        report += "| 方法 | 估算查询次数 | 平均L0 | 查询效率 (查询/像素) |\n"
        report += "|------|-------------|--------|---------------------|\n"
        
        for method in self.methods:
            if method in query_stats[model]:
                stats = query_stats[model][method]
                name = self.method_names[method]
                queries = stats['estimated_queries']
                l0 = stats['avg_l0']
                efficiency = queries / l0 if l0 > 0 else 0
                report += f"| {name} | ~{queries} | {l0:.2f} | {efficiency:.1f} |\n"
        
        report += "\n---\n\n## 3. 效率权衡分析\n\n"
        report += "### 3.1 ASR vs 时间\n\n"
        report += "| 方法 | ASR | 平均时间 | 效率分数 (ASR/Time) |\n"
        report += "|------|-----|---------|--------------------|\n"
        
        for d in tradeoff_data:
            efficiency_score = d['asr'] / d['avg_time'] if d['avg_time'] > 0 else 0
            report += f"| {d['method']} | {d['asr']:.1f}% | {d['avg_time']:.3f}s | {efficiency_score:.1f} |\n"
        
        # 找出最高效的方法
        best_efficiency = max(tradeoff_data, key=lambda x: x['asr'] / x['avg_time'] if x['avg_time'] > 0 else 0)
        
        report += f"\n**最高效方法:** {best_efficiency['method']} "
        report += f"(效率分数: {best_efficiency['asr'] / best_efficiency['avg_time']:.1f})\n\n"
        
        report += "### 3.2 L0 vs 时间\n\n"
        report += "| 方法 | 平均L0 | 平均时间 | 稀疏性效率 (L0/Time) |\n"
        report += "|------|--------|---------|---------------------|\n"
        
        for d in tradeoff_data:
            sparsity_efficiency = d['avg_l0'] / d['avg_time'] if d['avg_time'] > 0 else 0
            report += f"| {d['method']} | {d['avg_l0']:.2f} | {d['avg_time']:.3f}s | {sparsity_efficiency:.2f} |\n"
        
        report += "\n---\n\n## 4. 关键发现\n\n"
        
        fastest = min(tradeoff_data, key=lambda x: x['avg_time'])
        slowest = max(tradeoff_data, key=lambda x: x['avg_time'])
        
        report += f"1. **速度范围：** {fastest['avg_time']:.3f}s ({fastest['method']}) "
        report += f"至 {slowest['avg_time']:.3f}s ({slowest['method']})\n"
        report += f"2. **速度差异：** {slowest['avg_time'] / fastest['avg_time']:.1f}倍\n"
        report += f"3. **最佳效率：** {best_efficiency['method']} "
        report += f"({best_efficiency['asr']:.1f}% ASR, {best_efficiency['avg_time']:.3f}s)\n\n"
        
        report += "### 4.1 方法选择建议（基于效率）\n\n"
        report += "| 场景 | 推荐方法 | 理由 |\n"
        report += "|------|---------|------|\n"
        report += f"| 实时攻击 | {fastest['method']} | 最快 ({fastest['avg_time']:.3f}s) |\n"
        report += f"| 离线攻击 | {best_efficiency['method']} | 最佳效率 |\n"
        report += "| 高ASR需求 | JSMA | 高成功率 |\n"
        report += "| 极致稀疏 | SparseFool | 最小L0 |\n\n"
        
        report += "---\n\n## 5. 实践启示\n\n"
        report += "1. **黑盒攻击场景：** 查询次数受限时，选择查询效率高的方法（Greedy、PixelGrad）\n"
        report += "2. **白盒攻击场景：** 可以使用梯度信息，JSMA和SparseFool更合适\n"
        report += "3. **实时场景：** 选择最快的方法，牺牲一定成功率换取速度\n"
        report += "4. **评估场景：** 使用多种方法全面测试，平衡ASR、L0和时间\n\n"
        
        report += "---\n\n## 6. 可视化\n\n"
        report += "1. **time_comparison.pdf** - 3个模型的时间对比\n"
        report += "2. **efficiency_tradeoff.pdf** - ASR vs 时间散点图\n"
        report += "3. **time_l0_tradeoff.pdf** - L0 vs 时间权衡\n\n"
        
        report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # 保存报告
        report_file = self.output_dir / 'query_efficiency_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 分析报告已保存: {report_file}")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("查询效率分析")
       
        
        # 1. 加载数据
        self.load_all_data()
        
        # 2. 分析时间统计
        time_stats = self.analyze_time_statistics()
        
        # 3. 估算查询次数
        query_stats = self.calculate_query_estimates(time_stats)
        
        # 4. 分析效率权衡
        tradeoff_data = self.analyze_efficiency_tradeoff(time_stats)
        
        # 5. 生成可视化
        self.generate_visualizations(time_stats, tradeoff_data)
        
        # 6. 生成报告
        self.generate_report(time_stats, query_stats, tradeoff_data)
        
        print("查询效率分析完成！")
        
        print(f"\n 结果保存在: {self.output_dir}")
        print("\n生成的文件:")
        print("  1. query_efficiency_report.md - 完整分析报告")
        print("  2. time_comparison.pdf - 时间对比图")
        print("  3. efficiency_tradeoff.pdf - 效率权衡散点图")
        print("  4. time_l0_tradeoff.pdf - 时间-L0权衡图")

def main():
    analyzer = QueryEfficiencyAnalyzer()
    analyzer.run_complete_analysis()
    return 0

if __name__ == '__main__':
    exit(main())















