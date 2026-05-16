"""
Count dataset images in each cluster
Shows how many images per cluster before augmentation
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict


def count_cluster_data(partition_dir):
    """Count images per cluster"""
    
    if not os.path.exists(partition_dir):
        print(f"❌ Partition directory not found: {partition_dir}")
        return
    
    # Get train/val paths
    train_dir = os.path.join(partition_dir, 'train', 'images')
    val_dir = os.path.join(partition_dir, 'val', 'images')
    
    if not os.path.exists(train_dir):
        print(f"❌ Train images not found: {train_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 CHARM Dataset Statistics (Before Augmentation)")
    print(f"{'='*70}\n")
    
    # Count per cluster
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    
    # Train
    if os.path.exists(train_dir):
        for cluster in os.listdir(train_dir):
            cluster_path = os.path.join(train_dir, cluster)
            if os.path.isdir(cluster_path):
                count = len([f for f in os.listdir(cluster_path) 
                           if f.endswith(('.jpg', '.png'))])
                train_counts[cluster] = count
    
    # Val
    if os.path.exists(val_dir):
        for cluster in os.listdir(val_dir):
            cluster_path = os.path.join(val_dir, cluster)
            if os.path.isdir(cluster_path):
                count = len([f for f in os.listdir(cluster_path) 
                           if f.endswith(('.jpg', '.png'))])
                val_counts[cluster] = count
    
    # Print summary
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_clusters = len(train_counts)
    
    print(f"📁 TRAIN SET: {total_train:,} images in {total_clusters} clusters")
    print(f"📁 VAL SET:   {total_val:,} images")
    print(f"📁 TOTAL (ORIGINAL): {total_train + total_val:,} images\n")
    
    # Augmentation calculation
    print(f"{'='*70}")
    print(f"🔄 AUGMENTATION CALCULATION")
    print(f"{'='*70}\n")
    
    for top_k in [1, 2, 3, 5]:
        total_augmented = total_train + (total_train * top_k)
        print(f"With top_k_hard={top_k}:")
        print(f"  Original train: {total_train:,}")
        print(f"  + Variants:     {total_train * top_k:,} ({top_k}x)")
        print(f"  = Total train:  {total_augmented:,}")
        print(f"  Expansion:      {(total_augmented/total_train):.1f}x")
        print()
    
    # Per-cluster breakdown
    print(f"{'='*70}")
    print(f"📋 PER-CLUSTER BREAKDOWN")
    print(f"{'='*70}\n")
    print(f"{'Cluster':<15} {'Train':<10} {'Val':<10} {'Total':<10}")
    print(f"{'-'*50}")
    
    for cluster in sorted(train_counts.keys()):
        train_c = train_counts.get(cluster, 0)
        val_c = val_counts.get(cluster, 0)
        total_c = train_c + val_c
        print(f"{cluster:<15} {train_c:<10} {val_c:<10} {total_c:<10}")
    
    print(f"{'-'*50}")
    print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_train + total_val:<10}")
    
    print(f"\n{'='*70}")
    print(f"💡 NOTE: With top_k_hard=3 (default):")
    print(f"   {total_train:,} originals → {total_train * 4:,} total train images")
    print(f"   (1 original + 3 augmented variants per image)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count cluster dataset statistics')
    parser.add_argument('--partition', type=str, default='data/COD10K_Partitioning',
                        help='Path to partition directory')
    
    args = parser.parse_args()
    count_cluster_data(args.partition)
