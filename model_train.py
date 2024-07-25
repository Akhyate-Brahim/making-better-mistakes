import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from better_mistakes.model.init import PlainNet
from better_mistakes.trees import load_hierarchy, load_distances, get_weighting
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss
from better_mistakes.model.labels import make_all_soft_labels
import os
import json
import argparse
import numpy as np
import logging

def setup_logger():
    logger = logging.getLogger('CIFAR100_Training')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class ReorderedDataset(Dataset):
    def __init__(self, original_dataset, new_order):
        self.original_dataset = original_dataset
        self.new_order = new_order
        self.old_to_new = {old: new for new, old in enumerate(new_order)}

    def __getitem__(self, index):
        img, old_label = self.original_dataset[index]
        new_label = self.old_to_new[self.original_dataset.classes[old_label]]
        return img, new_label

    def __len__(self):
        return len(self.original_dataset)

def train(model, train_loader, criterion, optimizer, device, soft_labels=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if soft_labels is not None:
            target_distribution = soft_labels[labels]
            loss = criterion(outputs.log_softmax(dim=1), target_distribution)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device, soft_labels=None):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if soft_labels is not None:
                target_distribution = soft_labels[labels]
                loss = criterion(outputs.log_softmax(dim=1), target_distribution)
            else:
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / total
    return val_loss, val_accuracy

def calculate_hierarchical_metrics(outputs, targets, distances, classes):
    topK_to_consider = (1, 5, 10, 20, 100)
    max_dist = max(distances.distances.values())
    batch_size = outputs.size(0)

    _, topK_predicted_classes = outputs.topk(max(topK_to_consider), dim=1)
    topK_predicted_classes = topK_predicted_classes.cpu().numpy()
    targets = targets.cpu().numpy()

    topK_hdist = np.zeros((batch_size, max(topK_to_consider)))
    for i in range(batch_size):
        for j in range(max(topK_to_consider)):
            class_idx_ground_truth = targets[i]
            class_idx_predicted = topK_predicted_classes[i][j]
            topK_hdist[i, j] = distances[(classes[class_idx_predicted], classes[class_idx_ground_truth])]

    # Calculate mistakes
    mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
    topK_hdist_mistakes = topK_hdist[mistakes_ids, :]

    topK_hsimilarity = 1 - topK_hdist / max_dist

    best_hier_similarities = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort([distances[(classes[i], classes[j])] for j in range(len(classes))]) / max_dist

    hdist_avg = [np.mean(topK_hdist[:, :k]) for k in topK_to_consider]
    hdist_top = [np.mean([np.min(topK_hdist[b, :k]) for b in range(batch_size)]) for k in topK_to_consider]
    hdist_mistakes = [np.sum(topK_hdist_mistakes[:, :k]) / (len(mistakes_ids) * k) if len(mistakes_ids) > 0 else 0 for k in topK_to_consider]
    hprecision = [np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k]) for k in topK_to_consider]
    hmAP = [np.mean([np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k]) for k in range(1, k+1)]) for k in topK_to_consider]

    return hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, len(mistakes_ids)

def test(model, test_loader, criterion, device, distances, classes, soft_labels=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []
    all_predictions = []  # New list to store predictions

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if soft_labels is not None:
                target_distribution = soft_labels[targets]
                loss = criterion(outputs.log_softmax(dim=1), target_distribution)
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_outputs.append(outputs)
            all_targets.append(targets)
            all_predictions.append(predicted)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes = calculate_hierarchical_metrics(all_outputs, all_targets, distances, classes)

    # Save data for confusion matrix
    confusion_matrix_data = {
        'true_labels': all_targets.cpu().numpy().tolist(),
        'predictions': all_predictions.cpu().numpy().tolist()
    }

    confusion_matrix_path = os.path.join(opts.out_folder, 'confusion_matrix_data.json')
    with open(confusion_matrix_path, 'w') as f:
        json.dump(confusion_matrix_data, f)


    return test_loss, test_accuracy, hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes, total

def main(opts):
    logger = setup_logger()
    logger.info("Starting the training process")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    full_train_dataset = datasets.ImageFolder(root=os.path.join(opts.data_path, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(opts.data_path, 'val'), transform=transform_test)

    hierarchy = load_hierarchy(opts.data, opts.data_dir)
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)

    hierarchy_leaves = [str(leaf) for leaf in hierarchy.leaves()]

    original_to_new = {cls: idx for idx, cls in enumerate(hierarchy_leaves)}

    full_train_dataset = ReorderedDataset(full_train_dataset, hierarchy_leaves)
    val_dataset = ReorderedDataset(val_dataset, hierarchy_leaves)

    train_size = int(0.8 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)

    model = PlainNet(output_dim=len(hierarchy_leaves))
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.info(f"Classes (first 10): {hierarchy_leaves[:10]}")
    logger.info(f"Hierarchy leaves (first 10): {hierarchy_leaves[:10]}")

    assert hierarchy_leaves == hierarchy_leaves, "Class order does not match hierarchy leaf order"

    soft_labels = None
    if opts.loss == "soft-labels":
        soft_labels = make_all_soft_labels(distances, hierarchy_leaves, opts.beta)
        soft_labels = torch.FloatTensor(soft_labels).to(device)

    if opts.loss == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        criterion = HierarchicalCrossEntropyLoss(hierarchy, hierarchy_leaves, weights)
    elif opts.loss == "soft-labels":
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unknown loss function: {opts.loss}")

    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr)

    for epoch in range(opts.epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, soft_labels)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, soft_labels)

        logger.info(f'Epoch [{epoch+1}/{opts.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_accuracy:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_accuracy:.2f}%')

    logger.info('Training completed.')

    test_loss, test_accuracy, hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes, total = test(model, test_loader, criterion, device, distances, hierarchy_leaves, soft_labels)

    logger.info("\nFinal Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    logger.info(f"Number of mistakes: {num_mistakes}/{total}")
    logger.info(f"Hierarchical Distance Avg: {hdist_avg}")
    logger.info(f"Hierarchical Distance Top: {hdist_top}")
    logger.info(f"Hierarchical Distance Mistakes: {hdist_mistakes}")
    logger.info(f"Hierarchical Precision: {hprecision}")
    logger.info(f"Hierarchical mAP: {hmAP}")

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "num_mistakes": num_mistakes,
        "total_samples": total,
        "hierarchical_distance_avg": hdist_avg,
        "hierarchical_distance_top": hdist_top,
        "hierarchical_distance_mistakes": hdist_mistakes,
        "hierarchical_precision": hprecision,
        "hierarchical_mAP": hmAP
    }

    with open(os.path.join(opts.out_folder, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(opts.out_folder, 'opts.json'), 'w') as f:
        json.dump(vars(opts), f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-100 Training and Testing')
    parser.add_argument('--data_path', type=str, default='./cifar-100', help='path to dataset')
    parser.add_argument('--data_dir', type=str, default='./data/', help='path to supplementary data')
    parser.add_argument('--out_folder', type=str, default='./experiment', help='folder to save outputs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--loss', type=str, default='cross-entropy',
                        choices=['cross-entropy', 'hierarchical-cross-entropy', 'soft-labels'],
                        help='loss function')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha parameter for hierarchical loss')
    parser.add_argument('--beta', type=float, default=0, help='beta parameter for soft labels')
    opts = parser.parse_args()

    os.makedirs(opts.out_folder, exist_ok=True)

    main(opts)