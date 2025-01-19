import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import defaultdict


# Create folders for graphs if they don't exist
os.makedirs("nvidia", exist_ok=True)
os.makedirs("amd", exist_ok=True)


# Read JSON file
with open('benchmark.json', 'r') as file:
    data = json.load(file)


def create_graphs(device_data, device_type):
    # Group data by algorithm
    algorithms = defaultdict(list)
    
    for entry in device_data:
        # Extract base algorithm from name
        algo_base = entry['name'].split(':')[0].strip()
        algorithms[algo_base].append(entry)
    
    # For each algorithm, create graphs
    for algo, measurements in algorithms.items():
        # Create DataFrame for the algorithm
        df = pd.DataFrame(measurements)
        
        # Average performance graph
        plt.figure(figsize=(12, 6))
        perf_means = df.groupby('name')['perform'].mean()
        perf_std = df.groupby('name')['perform'].std()
        
        # Create bar plot manually
        bars = plt.bar(range(len(perf_means)), perf_means.values)
        plt.xticks(range(len(perf_means)), perf_means.index, rotation=45, ha='right')
        
        # Add error bars
        plt.errorbar(
            range(len(perf_means)),
            perf_means.values,
            yerr=perf_std.values,
            fmt='none',
            color='black',
            capsize=5)
        
        plt.title(f'Performance Comparison - {algo}')
        plt.xlabel('Implementation')
        plt.ylabel('Performance')
        plt.tight_layout()
        plt.savefig(f'{device_type}/{algo.replace(" ", "_")}_comparison.png')
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='name', y='perform')
        plt.title(f'Performance Distribution - {algo}')
        plt.xlabel('Implementation')
        plt.ylabel('Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{device_type}/{algo.replace(" ", "_")}_boxplot.png')
        plt.close()
        
        # Violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='name', y='perform')
        plt.title(f'Performance Distribution (Violin) - {algo}')
        plt.xlabel('Implementation')
        plt.ylabel('Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{device_type}/{algo.replace(" ", "_")}_violin.png')
        plt.close()


# Create graphs for NVIDIA
if data['nvidia']:
    create_graphs(data['nvidia'], 'nvidia')

# Create graphs for AMD
if data['amd']:
    create_graphs(data['amd'], 'amd')

print("Graphs successfully generated!")
