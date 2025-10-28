import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from PIL import Image, ImageDraw
import seaborn as sns
from pathlib import Path


class BDD100KParser:
    def __init__(self, base_path):
        """
        Initialize the BDD100K parser.

        Args:
            base_path (str): Base path to the BDD100K dataset
        """
        self.base_path = base_path
        self.images_path = os.path.join(
            base_path, "bdd100k_images_100k/bdd100k/images", "100k"
        )
        self.labels_path = os.path.join(
            base_path, "bdd100k_labels_release/bdd100k", "labels"
        )
        self.train_json = os.path.join(
            self.labels_path, "bdd100k_labels_images_train.json"
        )
        self.val_json = os.path.join(self.labels_path, "bdd100k_labels_images_val.json")

        # Load annotations
        self.train_annotations = self._load_annotations(self.train_json)
        self.val_annotations = self._load_annotations(self.val_json)

        # Define class names
        self.class_names = [
            "car",
            "traffic sign",
            "traffic light",
            "person",
            "truck",
            "bus",
            "bike",
            "rider",
            "motor",
            "train",
        ]

    def _load_annotations(self, json_path):
        """Load annotations from a JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    def get_class_distribution(self, split="train"):
        """
        Get distribution of object classes in the specified split.

        Args:
            split (str): Dataset split ("train" or "val")

        Returns:
            Counter: Class distribution counts
        """
        annotations = (
            self.train_annotations if split == "train" else self.val_annotations
        )
        class_counts = Counter()

        for image_ann in annotations:
            if "labels" in image_ann:
                for label in image_ann["labels"]:
                    if "category" in label:
                        class_counts[label["category"]] += 1

        return class_counts

    def get_object_size_distribution(self, split="train"):
        """Get distribution of object sizes in the dataset."""
        annotations = (
            self.train_annotations if split == "train" else self.val_annotations
        )
        size_data = {cls: [] for cls in self.class_names}

        for image_ann in annotations:
            if "labels" in image_ann:
                for label in image_ann["labels"]:
                    if "category" in label and "box2d" in label:
                        category = label["category"]
                        if category in self.class_names:
                            box = label["box2d"]
                            width = box["x2"] - box["x1"]
                            height = box["y2"] - box["y1"]
                            area = width * height
                            size_data[category].append(area)

        return size_data

    def get_spatial_distribution(self, split="train", grid_size=(10, 10)):
        """Get spatial distribution of objects in the dataset."""
        annotations = (
            self.train_annotations if split == "train" else self.val_annotations
        )
        spatial_data = {cls: np.zeros(grid_size) for cls in self.class_names}

        for image_ann in annotations:
            if "labels" in image_ann:
                for label in image_ann["labels"]:
                    if "category" in label and "box2d" in label:
                        category = label["category"]
                        if category in self.class_names:
                            box = label["box2d"]
                            center_x = (box["x1"] + box["x2"]) / 2
                            center_y = (box["y1"] + box["y2"]) / 2

                            # Normalize to 0-1
                            norm_x = center_x / 1280  # BDD100K image width
                            norm_y = center_y / 720  # BDD100K image height

                            # Map to grid
                            grid_x = min(int(norm_x * grid_size[0]), grid_size[0] - 1)
                            grid_y = min(int(norm_y * grid_size[1]), grid_size[1] - 1)

                            spatial_data[category][grid_y, grid_x] += 1

        return spatial_data

    def get_attributes_distribution(self, split="train"):
        """Get distribution of weather, time of day, and scene attributes."""
        annotations = (
            self.train_annotations if split == "train" else self.val_annotations
        )
        weather_counts = Counter()
        timeofday_counts = Counter()
        scene_counts = Counter()

        for image_ann in annotations:
            if "attributes" in image_ann:
                attrs = image_ann["attributes"]
                if "weather" in attrs:
                    weather_counts[attrs["weather"]] += 1
                if "timeofday" in attrs:
                    timeofday_counts[attrs["timeofday"]] += 1
                if "scene" in attrs:
                    scene_counts[attrs["scene"]] += 1

        return {
            "weather": weather_counts,
            "timeofday": timeofday_counts,
            "scene": scene_counts,
        }

    def find_anomalous_samples(self, threshold=0.01):
        """Find images with rare object combinations or unusual characteristics."""
        # Implementation would look for rare class combinations,
        # extremely crowded scenes, or other anomalous patterns
        pass

    def visualize_sample(self, image_id, output_path=None):
        """Visualize a sample image with bounding boxes."""
        # Find image annotation
        image_ann = None
        for ann in self.train_annotations:
            if ann["name"] == image_id:
                image_ann = ann
                break

        if not image_ann:
            for ann in self.val_annotations:
                if ann["name"] == image_id:
                    image_ann = ann
                    break

        if not image_ann:
            return None

        # Determine image path
        split = "train" if image_ann in self.train_annotations else "val"
        image_path = os.path.join(self.images_path, split, image_id)

        # Draw bounding boxes
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        colors = {
            "car": (255, 0, 0),
            "traffic sign": (0, 255, 0),
            "traffic light": (0, 0, 255),
            "person": (255, 255, 0),
            "truck": (255, 0, 255),
            "bus": (0, 255, 255),
            "bike": (128, 0, 0),
            "rider": (0, 128, 0),
            "motor": (0, 0, 128),
            "train": (128, 128, 0),
        }

        for label in image_ann.get("labels", []):
            if "category" in label and "box2d" in label:
                category = label["category"]
                box = label["box2d"]

                if category in colors:
                    color = colors[category]
                    draw.rectangle(
                        [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
                        outline=color,
                        width=3,
                    )
                    draw.text((box["x1"], box["y1"] - 10), category, fill=color)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

        return image


if __name__ == "__main__":
    # Example code to analyze class distribution
    parser = BDD100KParser("assignment_data_bdd")
    train_class_dist = parser.get_class_distribution("train")
    val_class_dist = parser.get_class_distribution("val")

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(
        train_class_dist.keys(), [train_class_dist[k] for k in train_class_dist.keys()]
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Object Class Distribution in Training Set")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_frequency.png")

    size_data = parser.get_object_size_distribution("train")

    # Plot size distributions as box plots
    plt.figure(figsize=(14, 8))
    box_data = [size_data[cls] for cls in parser.class_names if size_data[cls]]
    plt.boxplot(box_data, labels=[cls for cls in parser.class_names if size_data[cls]])
    plt.xticks(rotation=45, ha="right")
    plt.title("Object Size Distribution by Class")
    plt.ylabel("Area (pixels²)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("object_distribution.png")

    spatial_data = parser.get_spatial_distribution("train")

    # Plot heatmaps for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, cls in enumerate(parser.class_names):
        sns.heatmap(spatial_data[cls], ax=axes[i], cmap="YlOrRd")
        axes[i].set_title(f"Spatial Distribution: {cls}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig("spatial_distribution.png")

    # Compare class distributions between train and validation sets
    train_dist = parser.get_class_distribution("train")
    val_dist = parser.get_class_distribution("val")

    # Normalize counts by dataset size
    train_size = sum(train_dist.values())
    val_size = sum(val_dist.values())

    train_norm = {k: v / train_size for k, v in train_dist.items()}
    val_norm = {k: v / val_size for k, v in val_dist.items()}

    # Plot comparative distribution
    plt.figure(figsize=(14, 7))
    x = np.arange(len(train_norm))
    width = 0.35

    plt.bar(x - width / 2, list(train_norm.values()), width, label="Train")
    plt.bar(x + width / 2, list(val_norm.values()), width, label="Validation")
    plt.xticks(x, list(train_norm.keys()), rotation=45, ha="right")
    plt.title("Normalized Class Distribution: Train vs Validation")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.tight_layout()

    plt.savefig("class_distribution.png")

    # Analyze environmental attributes
    attr_dist = parser.get_attributes_distribution("train")

    # Plot weather distribution
    plt.figure(figsize=(10, 6))
    weather_counts = attr_dist["weather"]
    plt.bar(weather_counts.keys(), [weather_counts[k] for k in weather_counts.keys()])
    plt.xticks(rotation=45, ha="right")
    plt.title("Weather Distribution in Training Set")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("weather_distribution.png")

    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(parser.class_names), len(parser.class_names)))

    for image_ann in parser.train_annotations:
        if "labels" in image_ann:
            # Get unique classes in this image
            classes_in_image = set()
            for label in image_ann["labels"]:
                if "category" in label and label["category"] in parser.class_names:
                    classes_in_image.add(parser.class_names.index(label["category"]))

            # Update co-occurrence matrix
            for i in classes_in_image:
                for j in classes_in_image:
                    co_occurrence[i, j] += 1

    # Plot co-occurrence heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        co_occurrence,
        xticklabels=parser.class_names,
        yticklabels=parser.class_names,
        cmap="YlGnBu",
    )
    plt.title("Class Co-occurrence in Training Set")
    plt.tight_layout()
    plt.savefig("coocurance.png")

    # Find images with unusually high object counts
    object_counts = []
    for image_ann in parser.train_annotations:
        if "labels" in image_ann:
            count = len([l for l in image_ann["labels"] if "category" in l])
            object_counts.append((image_ann["name"], count))

    # Sort by count descending
    object_counts.sort(key=lambda x: x[1], reverse=True)
    most_crowded = object_counts[:10]  # Top 10 most crowded images
    # print("Most Crowded images:", most_crowded)

    crowded_samples = [name for name, count in object_counts[:5]]
    for sample in crowded_samples:
        parser.visualize_sample(sample, f"corwded_samples/crowded_{sample}.jpg")

    # Find samples with rare class combinations
    # For example, images containing both 'train' and 'car'
    train_car_samples = []
    for image_ann in parser.train_annotations:
        if "labels" in image_ann:
            categories = [
                label["category"]
                for label in image_ann["labels"]
                if "category" in label
            ]
            if "train" in categories and "car" in categories:
                train_car_samples.append(image_ann["name"])

    # Visualize these samples
    for sample in train_car_samples[:3]:
        parser.visualize_sample(sample, f"rare_samples/train_car_{sample}.jpg")
