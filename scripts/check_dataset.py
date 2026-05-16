"""
Check CHARM Dataset Integrity
Validates generated dataset structure and statistics
"""

import os
from glob import glob
import argparse


def check_charm_dataset(root):
    """Check dataset integrity and statistics"""
    
    print(f"\n📊 CHARM Dataset Integrity Check")
    print(f"{'='*60}")
    print(f"Dataset root: {root}\n")
    
    # Check structure
    train_imgs = sorted(glob(os.path.join(root, 'train', 'images', '*.jpg'))) + \
                 sorted(glob(os.path.join(root, 'train', 'images', '*.png')))
    train_labels = sorted(glob(os.path.join(root, 'train', 'labels', '*.txt')))
    
    val_imgs = sorted(glob(os.path.join(root, 'val', 'images', '*.jpg'))) + \
               sorted(glob(os.path.join(root, 'val', 'images', '*.png')))
    val_labels = sorted(glob(os.path.join(root, 'val', 'labels', '*.txt')))
    
    # Statistics
    print(f"📁 TRAIN SET")
    print(f"   Images: {len(train_imgs):,}")
    print(f"   Labels: {len(train_labels):,}")
    print(f"   Match: {'✅' if len(train_imgs) == len(train_labels) else '❌'}")
    
    print(f"\n📁 VAL SET")
    print(f"   Images: {len(val_imgs):,}")
    print(f"   Labels: {len(val_labels):,}")
    print(f"   Match: {'✅' if len(val_imgs) == len(val_labels) else '❌'}")
    
    print(f"\n📊 AUGMENTATION ANALYSIS")
    
    # Count originals vs augmented
    train_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in train_imgs}
    
    originals = len([b for b in train_basenames if '_charm_v' not in b])
    augmented = len([b for b in train_basenames if '_charm_v' in b])
    
    print(f"   Original images: {originals:,}")
    print(f"   Augmented images: {augmented:,}")
    
    if originals > 0:
        avg_variants = augmented / originals
        print(f"   Variants per original: {avg_variants:.1f}x")
    
    # Check mismatches
    print(f"\n🔍 VALIDATION")
    
    img_basenames_set = set(os.path.splitext(os.path.basename(p))[0] for p in train_imgs)
    label_basenames_set = set(os.path.splitext(os.path.basename(p))[0] for p in train_labels)
    
    missing_labels = img_basenames_set - label_basenames_set
    missing_images = label_basenames_set - img_basenames_set
    
    if missing_labels:
        print(f"   ❌ Images without labels: {len(missing_labels)}")
        print(f"      Samples: {list(missing_labels)[:3]}")
    else:
        print(f"   ✅ All train images have labels")
    
    if missing_images:
        print(f"   ❌ Labels without images: {len(missing_images)}")
        print(f"      Samples: {list(missing_images)[:3]}")
    else:
        print(f"   ✅ All train labels have images")
    
    # Check image sizes (sample)
    print(f"\n📐 IMAGE SIZES (sample 10)")
    sizes = set()
    for p in train_imgs[:10]:
        try:
            import cv2
            im = cv2.imread(p)
            if im is not None:
                sizes.add(im.shape[:2])
        except:
            pass
    
    if sizes:
        print(f"   Detected sizes: {sizes}")
    
    # Check data.yaml exists
    yaml_path = os.path.join(root, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"\n✅ data.yaml found")
        with open(yaml_path) as f:
            print(f"   Preview:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"     {line.rstrip()}")
    else:
        print(f"\n❌ data.yaml NOT found")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ CHARM dataset ready for training!" if len(train_imgs) == len(train_labels) and len(val_imgs) == len(val_labels) else "⚠️ Check issues above")
    print(f"   Total training images: {len(train_imgs):,}")
    print(f"   Total validation images: {len(val_imgs):,}")
    print(f"   Total: {len(train_imgs) + len(val_imgs):,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check CHARM Dataset Integrity')
    parser.add_argument('--dataset', type=str, default='data/COD10K-CHARM',
                        help='Path to CHARM dataset root')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
    else:
        check_charm_dataset(args.dataset)
