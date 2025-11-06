"""
Visualization module for RL NAS project.

This module provides comprehensive visualization tools for training progress,
architecture analysis, and results presentation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import torch
from pathlib import Path
import json


class RLNASVisualizer:
    """Main visualization class for RL NAS project."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with style."""
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_training_progress(
        self,
        metrics: Dict[str, List[float]],
        steps: List[int],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training progress metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Episode rewards
        axes[0, 0].plot(steps, metrics.get('episode_rewards', []), 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moving average
        if len(metrics.get('episode_rewards', [])) > 10:
            window_size = min(50, len(metrics['episode_rewards']) // 10)
            moving_avg = pd.Series(metrics['episode_rewards']).rolling(window_size).mean()
            axes[0, 0].plot(steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(steps, metrics.get('episode_lengths', []), 'g-', alpha=0.7)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curves
        if 'policy_loss' in metrics:
            axes[1, 0].plot(steps, metrics['policy_loss'], 'purple', alpha=0.7, label='Policy Loss')
        if 'value_loss' in metrics:
            axes[1, 0].plot(steps, metrics['value_loss'], 'orange', alpha=0.7, label='Value Loss')
        axes[1, 0].set_title('Loss Curves')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation rewards
        if 'eval_rewards' in metrics:
            axes[1, 1].plot(metrics['eval_steps'], metrics['eval_rewards'], 'ro-', markersize=4)
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Eval Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_architecture_analysis(
        self,
        architectures: List[List[int]],
        rewards: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot architecture analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Architecture Analysis', fontsize=16, fontweight='bold')
        
        # Architecture size vs reward
        arch_sizes = [sum(arch) for arch in architectures]
        axes[0, 0].scatter(arch_sizes, rewards, alpha=0.6, c=rewards, cmap='viridis')
        axes[0, 0].set_title('Architecture Size vs Reward')
        axes[0, 0].set_xlabel('Total Parameters (sum of layer sizes)')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of layers vs reward
        num_layers = [len(arch) for arch in architectures]
        axes[0, 1].scatter(num_layers, rewards, alpha=0.6, c=rewards, cmap='viridis')
        axes[0, 1].set_title('Number of Layers vs Reward')
        axes[0, 1].set_xlabel('Number of Layers')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Layer size distribution
        all_layer_sizes = []
        for arch in architectures:
            all_layer_sizes.extend(arch)
        
        axes[1, 0].hist(all_layer_sizes, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Layer Size Distribution')
        axes[1, 0].set_xlabel('Layer Size')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Top architectures
        top_indices = np.argsort(rewards)[-5:]  # Top 5
        top_archs = [architectures[i] for i in top_indices]
        top_rewards = [rewards[i] for i in top_indices]
        
        y_pos = np.arange(len(top_archs))
        axes[1, 1].barh(y_pos, top_rewards, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([str(arch) for arch in top_archs])
        axes[1, 1].set_title('Top 5 Architectures')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_comparison(
        self,
        results: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison between different algorithms."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        
        # Learning curves
        for algo_name, metrics in results.items():
            if 'episode_rewards' in metrics and 'steps' in metrics:
                # Smooth the curve
                rewards = metrics['episode_rewards']
                steps = metrics['steps']
                if len(rewards) > 10:
                    window_size = min(50, len(rewards) // 10)
                    smooth_rewards = pd.Series(rewards).rolling(window_size).mean()
                    axes[0, 0].plot(steps, smooth_rewards, label=algo_name, linewidth=2)
                else:
                    axes[0, 0].plot(steps, rewards, label=algo_name, linewidth=2)
        
        axes[0, 0].set_title('Learning Curves')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final performance
        final_rewards = []
        algo_names = []
        for algo_name, metrics in results.items():
            if 'episode_rewards' in metrics and metrics['episode_rewards']:
                final_rewards.append(np.mean(metrics['episode_rewards'][-10:]))  # Last 10 episodes
                algo_names.append(algo_name)
        
        if final_rewards:
            bars = axes[0, 1].bar(algo_names, final_rewards, alpha=0.7)
            axes[0, 1].set_title('Final Performance')
            axes[0, 1].set_ylabel('Average Reward (Last 10 Episodes)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, final_rewards):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Sample efficiency
        for algo_name, metrics in results.items():
            if 'episode_rewards' in metrics and 'steps' in metrics:
                rewards = metrics['episode_rewards']
                steps = metrics['steps']
                
                # Find steps to reach 50% of max reward
                max_reward = max(rewards)
                target_reward = 0.5 * max_reward
                
                for i, reward in enumerate(rewards):
                    if reward >= target_reward:
                        axes[1, 0].bar(algo_name, steps[i], alpha=0.7, label=algo_name)
                        break
        
        axes[1, 0].set_title('Sample Efficiency (Steps to 50% Max Reward)')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stability (reward variance)
        stability_scores = []
        for algo_name, metrics in results.items():
            if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 10:
                rewards = metrics['episode_rewards'][-20:]  # Last 20 episodes
                stability = 1.0 / (np.std(rewards) + 1e-8)  # Higher is more stable
                stability_scores.append(stability)
            else:
                stability_scores.append(0)
        
        bars = axes[1, 1].bar(algo_names, stability_scores, alpha=0.7)
        axes[1, 1].set_title('Stability (1 / Reward Variance)')
        axes[1, 1].set_ylabel('Stability Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(
        self,
        training_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress', 'Architecture Analysis', 
                          'Reward Distribution', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training progress
        if 'episode_rewards' in training_data and 'steps' in training_data:
            fig.add_trace(
                go.Scatter(
                    x=training_data['steps'],
                    y=training_data['episode_rewards'],
                    mode='lines',
                    name='Episode Rewards',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Architecture analysis
        if 'architectures' in training_data and 'rewards' in training_data:
            arch_sizes = [sum(arch) for arch in training_data['architectures']]
            fig.add_trace(
                go.Scatter(
                    x=arch_sizes,
                    y=training_data['rewards'],
                    mode='markers',
                    name='Architecture Performance',
                    marker=dict(
                        size=8,
                        color=training_data['rewards'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=1, col=2
            )
        
        # Reward distribution
        if 'rewards' in training_data:
            fig.add_trace(
                go.Histogram(
                    x=training_data['rewards'],
                    name='Reward Distribution',
                    nbinsx=20
                ),
                row=2, col=1
            )
        
        # Performance metrics
        if 'metrics' in training_data:
            metrics = training_data['metrics']
            metric_names = list(metrics.keys())
            metric_values = [np.mean(values) if isinstance(values, list) else values 
                           for values in metrics.values()]
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Average Metrics'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="RL NAS Training Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Steps", row=1, col=1)
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        
        fig.update_xaxes(title_text="Architecture Size", row=1, col=2)
        fig.update_yaxes(title_text="Reward", row=1, col=2)
        
        fig.update_xaxes(title_text="Reward", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Metrics", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_architecture_heatmap(
        self,
        architectures: List[List[int]],
        rewards: List[float],
        max_layers: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot architecture performance heatmap."""
        # Create heatmap data
        layer_sizes = [32, 64, 128, 256, 512]
        heatmap_data = np.zeros((max_layers, len(layer_sizes)))
        
        for arch, reward in zip(architectures, rewards):
            for i, layer_size in enumerate(arch[:max_layers]):
                if layer_size in layer_sizes:
                    size_idx = layer_sizes.index(layer_size)
                    heatmap_data[i, size_idx] += reward
        
        # Normalize by count
        count_data = np.zeros_like(heatmap_data)
        for arch in architectures:
            for i, layer_size in enumerate(arch[:max_layers]):
                if layer_size in layer_sizes:
                    size_idx = layer_sizes.index(layer_size)
                    count_data[i, size_idx] += 1
        
        # Avoid division by zero
        heatmap_data = np.divide(heatmap_data, count_data, 
                                out=np.zeros_like(heatmap_data), 
                                where=count_data!=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(layer_sizes)))
        ax.set_xticklabels(layer_sizes)
        ax.set_yticks(range(max_layers))
        ax.set_yticklabels([f'Layer {i+1}' for i in range(max_layers)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Reward', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(max_layers):
            for j in range(len(layer_sizes)):
                if count_data[i, j] > 0:
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}\n({int(count_data[i, j])})',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Architecture Performance Heatmap')
        ax.set_xlabel('Layer Size')
        ax.set_ylabel('Layer Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_training_summary(
        self,
        results: Dict[str, Any],
        save_path: str
    ):
        """Save comprehensive training summary."""
        summary_path = Path(save_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary report
        summary = {
            "training_summary": {
                "total_steps": results.get("total_steps", 0),
                "total_episodes": results.get("total_episodes", 0),
                "final_reward": results.get("final_reward", 0),
                "best_reward": max(results.get("episode_rewards", [0])),
                "average_reward": np.mean(results.get("episode_rewards", [0])),
                "std_reward": np.std(results.get("episode_rewards", [0]))
            },
            "architecture_analysis": {
                "total_architectures": len(results.get("architectures", [])),
                "unique_architectures": len(set(str(arch) for arch in results.get("architectures", []))),
                "average_layers": np.mean([len(arch) for arch in results.get("architectures", [])]),
                "average_size": np.mean([sum(arch) for arch in results.get("architectures", [])])
            },
            "top_architectures": []
        }
        
        # Add top architectures
        if "architectures" in results and "rewards" in results:
            arch_reward_pairs = list(zip(results["architectures"], results["rewards"]))
            arch_reward_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (arch, reward) in enumerate(arch_reward_pairs[:5]):
                summary["architecture_analysis"]["top_architectures"].append({
                    "rank": i + 1,
                    "architecture": arch,
                    "reward": reward,
                    "size": sum(arch),
                    "layers": len(arch)
                })
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Training summary saved to: {summary_path}")


def create_visualization_report(
    results_path: str,
    output_dir: str = "./visualizations"
):
    """Create comprehensive visualization report from results."""
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RLNASVisualizer()
    
    # Create visualizations
    print("Creating training progress plot...")
    if "episode_rewards" in results and "steps" in results:
        fig1 = visualizer.plot_training_progress(
            {"episode_rewards": results["episode_rewards"]},
            results["steps"],
            save_path=str(output_path / "training_progress.png")
        )
        plt.close(fig1)
    
    print("Creating architecture analysis...")
    if "architectures" in results and "rewards" in results:
        fig2 = visualizer.plot_architecture_analysis(
            results["architectures"],
            results["rewards"],
            save_path=str(output_path / "architecture_analysis.png")
        )
        plt.close(fig2)
    
    print("Creating interactive dashboard...")
    fig3 = visualizer.create_interactive_dashboard(
        results,
        save_path=str(output_path / "interactive_dashboard.html")
    )
    
    print("Creating architecture heatmap...")
    if "architectures" in results and "rewards" in results:
        fig4 = visualizer.plot_architecture_heatmap(
            results["architectures"],
            results["rewards"],
            save_path=str(output_path / "architecture_heatmap.png")
        )
        plt.close(fig4)
    
    print("Saving training summary...")
    visualizer.save_training_summary(
        results,
        str(output_path / "training_summary.json")
    )
    
    print(f"Visualization report created in: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualization report")
    parser.add_argument("--results", type=str, required=True,
                      help="Path to results JSON file")
    parser.add_argument("--output", type=str, default="./visualizations",
                      help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    create_visualization_report(args.results, args.output)
