#!/usr/bin/env python3
"""
Parse BBBC041 Malaria Dataset Annotations

Reads the JSON annotation files and extracts:
- Image information (file paths, dimensions)
- Cell bounding boxes (coordinates)
- Cell categories (RBC, leukocyte, parasites)
- Dataset statistics (cell counts per class)

Output Format:
- Parsed annotations saved to data/bbbc041_malaria/parsed/
- Statistics summary saved to data/bbbc041_malaria/dataset_stats.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


def parse_json_annotations(json_path: Path) -> Tuple[List[Dict], Dict]:
    """
    Parse BBBC041 JSON annotation file.
    
    BBBC041 format (list of entries):
    [
        {
            "image": {
                "pathname": "/images/uuid.png",
                "shape": {"r": height, "c": width, "channels": 3}
            },
            "objects": [
                {
                    "bounding_box": {
                        "minimum": {"r": y1, "c": x1},
                        "maximum": {"r": y2, "c": x2}
                    },
                    "category": "red blood cell"
                }
            ]
        }
    ]
    
    Returns:
        Tuple of (parsed_annotations, statistics)
    """
    print(f"\n📖 Parsing {json_path.name}...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        print(f"❌ UTF-8 decode error: {e}")
        return [], {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return [], {}
    
    # Validate it's a list
    if not isinstance(data, list):
        print(f"❌ Expected list, got {type(data)}")
        return [], {}
    
    # Parse entries
    parsed = []
    class_counts = Counter()
    
    for entry in data:
        # Extract image info
        img_info = entry.get('image', {})
        filename = img_info.get('pathname', '').split('/')[-1]  # Extract just filename
        shape = img_info.get('shape', {})
        height = shape.get('r', 0)
        width = shape.get('c', 0)
        
        # Extract objects (cells)
        cells = []
        for obj in entry.get('objects', []):
            category = obj.get('category', 'unknown')
            bbox_dict = obj.get('bounding_box', {})
            
            # Extract coordinates (BBBC041 uses r=row=y, c=col=x)
            min_coords = bbox_dict.get('minimum', {})
            max_coords = bbox_dict.get('maximum', {})
            
            y1 = min_coords.get('r', 0)
            x1 = min_coords.get('c', 0)
            y2 = max_coords.get('r', 0)
            x2 = max_coords.get('c', 0)
            
            # Convert to [x, y, width, height] format (COCO-style)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            cells.append({
                'bbox': bbox,
                'bbox_xyxy': [x1, y1, x2, y2],  # Also keep xyxy for reference
                'category': category,
                'area': (x2 - x1) * (y2 - y1)
            })
            
            class_counts[category] += 1
        
        parsed.append({
            'filename': filename,
            'width': width,
            'height': height,
            'cells': cells
        })
    
    stats = {
        'total_images': len(parsed),
        'total_cells': sum(class_counts.values()),
        'class_distribution': dict(class_counts)
    }
    
    print(f"✅ Parsed {stats['total_images']} images, {stats['total_cells']} cells")
    
    return parsed, stats


def save_parsed_annotations(parsed: List[Dict], output_path: Path):
    """Save parsed annotations to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    
    print(f"💾 Saved parsed annotations to {output_path}")


def main():
    """Parse BBBC041 annotations and save results."""
    
    # Paths
    data_dir = Path("data/bbbc041_malaria/malaria")
    output_dir = Path("data/bbbc041_malaria/parsed")
    
    # Annotation files
    train_json = data_dir / "training.json"
    test_json = data_dir / "test.json"
    
    # Check files exist
    if not train_json.exists():
        print(f"❌ Training annotations not found: {train_json}")
        return
    if not test_json.exists():
        print(f"❌ Test annotations not found: {test_json}")
        return
    
    print("=" * 70)
    print("BBBC041 ANNOTATION PARSER")
    print("=" * 70)
    
    # Parse training set
    train_parsed, train_stats = parse_json_annotations(train_json)
    save_parsed_annotations(train_parsed, output_dir / "training_parsed.json")
    
    # Parse test set
    test_parsed, test_stats = parse_json_annotations(test_json)
    save_parsed_annotations(test_parsed, output_dir / "test_parsed.json")
    
    # Combined statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Training Set:")
    print(f"   Images: {train_stats['total_images']}")
    print(f"   Cells: {train_stats['total_cells']}")
    print(f"   Class distribution:")
    for category, count in sorted(train_stats['class_distribution'].items()):
        pct = count / train_stats['total_cells'] * 100
        print(f"      {category:20s}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\n📊 Test Set:")
    print(f"   Images: {test_stats['total_images']}")
    print(f"   Cells: {test_stats['total_cells']}")
    print(f"   Class distribution:")
    for category, count in sorted(test_stats['class_distribution'].items()):
        pct = count / test_stats['total_cells'] * 100
        print(f"      {category:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Combined stats
    total_images = train_stats['total_images'] + test_stats['total_images']
    total_cells = train_stats['total_cells'] + test_stats['total_cells']
    
    # Merge class distributions
    combined_classes = Counter()
    for cat, count in train_stats['class_distribution'].items():
        combined_classes[cat] += count
    for cat, count in test_stats['class_distribution'].items():
        combined_classes[cat] += count
    
    print(f"\n📊 Combined Dataset:")
    print(f"   Total images: {total_images}")
    print(f"   Total cells: {total_cells}")
    print(f"   Class distribution:")
    for category, count in sorted(combined_classes.items()):
        pct = count / total_cells * 100
        print(f"      {category:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Save combined statistics
    combined_stats = {
        'training': train_stats,
        'test': test_stats,
        'combined': {
            'total_images': total_images,
            'total_cells': total_cells,
            'class_distribution': dict(combined_classes)
        }
    }
    
    stats_path = Path("data/bbbc041_malaria/dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(combined_stats, f, indent=2)
    
    print(f"\n💾 Saved statistics to {stats_path}")
    print("\n✅ Parsing complete!")


if __name__ == "__main__":
    main()
