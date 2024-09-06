# Standard library imports
import argparse
import json
import logging
import os

# Third-party imports
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50

# Local imports
from better_mistakes.model.init import PlainNet
from better_mistakes.model.labels import make_all_soft_labels
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss
from better_mistakes.trees import load_hierarchy, load_distances, get_weighting
from custom_loss import SimpleHierarchyLoss
from models import *

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

def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, soft_labels=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if soft_labels is not None:
            target_distribution = soft_labels[labels]
            loss = criterion(outputs.log_softmax(dim=1), target_distribution)
        else:
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Step the scheduler
        scheduler.step(epoch + batch_idx / len(train_loader))

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
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
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

def test(model, test_loader, criterion, device, distances, classes, soft_labels=None, calculate_coarse_acc=False):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    all_predictions = []

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

            all_outputs.append(outputs)
            all_targets.append(targets)
            all_predictions.append(predicted)

    test_loss /= len(test_loader.dataset)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    correct = all_predictions.eq(all_targets).sum().item()
    total = all_targets.size(0)
    test_accuracy = 100. * correct / total

    coarse_accuracy = None
    if calculate_coarse_acc:
        correct_coarse = (all_predictions // 5).eq(all_targets // 5)
        coarse_accuracy = correct_coarse.float().mean().item() * 100

    hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes = calculate_hierarchical_metrics(all_outputs, all_targets, distances, classes)

    # Save data for confusion matrix
    confusion_matrix_data = {
        'true_labels': all_targets.cpu().numpy().tolist(),
        'predictions': all_predictions.cpu().numpy().tolist()
    }

    confusion_matrix_path = os.path.join(opts.out_folder, 'confusion_matrix_data.json')
    with open(confusion_matrix_path, 'w') as f:
        json.dump(confusion_matrix_data, f)

    return test_loss, test_accuracy, hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes, len(test_loader.dataset), coarse_accuracy

def test_simple_hierarchy(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_targets.append(targets)

    test_loss /= len(test_loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    fine_accuracy, coarse_accuracy, hierarchical_precision, hierarchical_similarity = calculate_simple_hierarchical_metrics(all_outputs, all_targets)

    return test_loss, fine_accuracy, coarse_accuracy, hierarchical_precision, hierarchical_similarity
def save_epoch_info(epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr, out_folder):
    epoch_info = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'learning_rate': lr
    }
    with open(os.path.join(out_folder, f'epoch_{epoch+1}_info.json'), 'w') as f:
        json.dump(epoch_info, f, indent=4)

def setup_data_loaders(opts):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
    if not opts.calculate_coarse_acc:
        full_train_dataset = ReorderedDataset(full_train_dataset, hierarchy_leaves)
        val_dataset = ReorderedDataset(val_dataset, hierarchy_leaves)

    train_size = int(0.8 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)

    return train_loader, val_loader, test_loader, hierarchy, distances, hierarchy_leaves

def create_model(opts, num_classes):
    model_classes = {
        'plainnet': lambda: PlainNet(output_dim=num_classes),
        'resnet50': lambda: resnet50(pretrained=False),
        'VGG19': VGG,
        'ResNet18': ResNet18,
        'PreActResNet18': PreActResNet18,
        'GoogLeNet': GoogLeNet,
        'DenseNet121': DenseNet121,
        'ResNeXt29_2x64d': ResNeXt29_2x64d,
        'MobileNet': MobileNet,
        'MobileNetV2': MobileNetV2,
        'DPN92': DPN92,
        'ShuffleNetG2': ShuffleNetG2,
        'SENet18': SENet18,
        'ShuffleNetV2': lambda: ShuffleNetV2(1),
        'EfficientNetB0': EfficientNetB0,
        'RegNetX_200MF': RegNetX_200MF,
        'SimpleDLA': SimpleDLA
    }

    model = model_classes[opts.architecture]()
    
    if opts.architecture == 'resnet50':
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def setup_criterion(opts, hierarchy, hierarchy_leaves):
    if opts.loss == "cross-entropy":
        return nn.CrossEntropyLoss()
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        return HierarchicalCrossEntropyLoss(hierarchy, hierarchy_leaves, weights)
    elif opts.loss == "soft-labels":
        return nn.KLDivLoss(reduction='batchmean')
    elif opts.loss == "simple-hierarchy":
        return SimpleHierarchyLoss(
            lambda_coarse=opts.lambda_coarse,
            lambda_fine=opts.lambda_fine,
            coarse_loss_start_epoch=opts.coarse_loss_start_epoch,
            coarse_loss_ramp_epochs=opts.coarse_loss_ramp_epochs,
            fine_loss_start_epoch=opts.fine_loss_start_epoch,
            fine_loss_ramp_epochs=opts.fine_loss_ramp_epochs
        )
    else:
        raise ValueError(f"Unknown loss function: {opts.loss}")

def setup_optimizer_and_scheduler(opts, model, train_loader):
    base_lr = 0.1
    min_lr = 1e-6
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)

    if opts.lr_schedule == 'cosine':
        T_0 = 12  # Initial cycle length
        T_mult = 2  # Factor to increase T_i after a restart
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr)
    elif opts.lr_schedule == 'cyclic':
        step_size = 10 * len(train_loader)
        scheduler = CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=base_lr,
            step_size_up=step_size,
            mode='triangular2',
            cycle_momentum=False
        )
    elif opts.lr_schedule == 'cutout':
        milestones = [60, 120, 160]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    return optimizer, scheduler


def log_test_results(logger, results):
    logger.info(f"Test Loss: {results['test_loss']:.4f}")
    
    if isinstance(results['test_accuracy'], dict):
        logger.info(f"Exact Match Ratio: {results['test_accuracy']['exact_match_ratio']:.2f}%")
        logger.info(f"Hamming Accuracy: {results['test_accuracy']['hamming_accuracy']:.2f}%")
        logger.info(f"Example-based Accuracy: {results['test_accuracy']['example_based_accuracy']:.2f}%")
        logger.info(f"Label-based Accuracy: {results['test_accuracy']['label_based_accuracy']:.2f}%")
    else:
        logger.info(f"Test Accuracy: {results['test_accuracy']:.2f}%")

    if 'coarse_accuracy' in results:
        logger.info(f"Coarse-grained Accuracy: {results['coarse_accuracy']:.2f}%")

    logger.info(f"Number of mistakes: {results['num_mistakes']}/{results['total_samples']}")
    logger.info(f"Hierarchical Distance Avg: {results['hierarchical_distance_avg']}")
    logger.info(f"Hierarchical Distance Top: {results['hierarchical_distance_top']}")
    logger.info(f"Hierarchical Distance Mistakes: {results['hierarchical_distance_mistakes']}")
    logger.info(f"Hierarchical Precision: {results['hierarchical_precision']}")
    logger.info(f"Hierarchical mAP: {results['hierarchical_mAP']}")

def main(opts):
    logger = setup_logger()
    logger.info("Starting the training process")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, hierarchy, distances, hierarchy_leaves = setup_data_loaders(opts)
    
    model = create_model(opts, len(hierarchy_leaves))
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = setup_criterion(opts, hierarchy, hierarchy_leaves)
    criterion = criterion.to(device)

    optimizer, scheduler = setup_optimizer_and_scheduler(opts, model, train_loader)

    soft_labels = None
    if opts.loss == "soft-labels":
        soft_labels = make_all_soft_labels(distances, hierarchy_leaves, opts.beta)
        soft_labels = torch.FloatTensor(soft_labels).to(device)

    epoch_results = []

    for epoch in range(opts.epochs):
        if opts.loss == "simple-hierarchy":
            criterion.update_epoch(epoch)
        
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, scheduler, device, epoch, soft_labels)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, soft_labels)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f'Epoch [{epoch+1}/{opts.epochs}] '
            f'Train Loss: {train_loss:.4f} '
            f'Train Acc: {train_accuracy:.2f}% '
            f'Val Loss: {val_loss:.4f} '
            f'Val Acc: {val_accuracy:.2f}% '
            f'LR: {current_lr:.6f} ')
        
        save_epoch_info(epoch, train_loss, train_accuracy, val_loss, val_accuracy, current_lr, opts.out_folder)

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        })

    # Save all epoch results
    with open(os.path.join(opts.out_folder, 'all_epoch_results.json'), 'w') as f:
        json.dump(epoch_results, f, indent=4)

    logger.info('Training completed.')

    # Final testing
    test_results = test(model, test_loader, criterion, device, distances, hierarchy_leaves, soft_labels, calculate_coarse_acc=opts.calculate_coarse_acc)
    
    logger.info("\nFinal Test Results:")
    log_test_results(logger, test_results)

    # Save final test results
    with open(os.path.join(opts.out_folder, 'final_test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    # Save training options
    with open(os.path.join(opts.out_folder, 'training_options.json'), 'w') as f:
        json.dump(vars(opts), f, indent=4)

    # Save model
    torch.save(model.state_dict(), os.path.join(opts.out_folder, 'final_model.pth'))

    logger.info(f"All results and model saved in {opts.out_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-100 Training and Testing')
    parser.add_argument('--data_path', type=str, default='./cifar-100', help='path to dataset')
    parser.add_argument('--data_dir', type=str, default='./data/', help='path to supplementary data')
    parser.add_argument('--out_folder', type=str, default='./experiment', help='folder to save outputs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=372, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--architecture', type=str, default='plainnet',
                        choices=['plainnet', 'resnet50', 'VGG19', 'ResNet18', 'PreActResNet18',
                                 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d', 'MobileNet',
                                 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2',
                                 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA'],
                        help='Network architecture to use')
    parser.add_argument('--loss', type=str, default='cross-entropy',
                        choices=['cross-entropy', 'hierarchical-cross-entropy', 'soft-labels', 'simple-hierarchy'],
                        help='loss function')
    parser.add_argument('--lambda_fine', type=float, default=1, help='penalty factor for subclasses')
    parser.add_argument('--lambda_coarse', type=float, default=1, help='penalty factor for superclasses')
    parser.add_argument('--coarse_loss_start_epoch', type=int, default=0, help='epoch to start coarse-grained loss')
    parser.add_argument('--coarse_loss_ramp_epochs', type=int, default=0, help='epochs to ramp up coarse-grained loss')
    parser.add_argument('--fine_loss_start_epoch', type=int, default=0, help='epoch to start fine-grained loss')
    parser.add_argument('--fine_loss_ramp_epochs', type=int, default=0, help='epochs to ramp up fine-grained loss')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha parameter for hierarchical loss')
    parser.add_argument('--beta', type=float, default=0, help='beta parameter for soft labels')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'cyclic', 'cutout'],
                        help='Learning rate schedule (cosine, cyclic, or cutout)')
    parser.add_argument('--calculate_coarse_acc', action='store_true', help='Calculate coarse-grained accuracy')
    opts = parser.parse_args()

    os.makedirs(opts.out_folder, exist_ok=True)

    main(opts)