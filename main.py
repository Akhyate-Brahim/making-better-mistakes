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
from better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from better_mistakes.util.label_embeddings import create_embedding_layer
from better_mistakes.model.losses import CosineLoss, RankingLoss, CosinePlusXentLoss
from better_mistakes.model.run_nn import run_nn
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

def main(opts):
    logger = setup_logger()
    logger.info("Starting the training process")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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
    if opts.calculate_coarse_acc == False:
        full_train_dataset = ReorderedDataset(full_train_dataset, hierarchy_leaves)
        val_dataset = ReorderedDataset(val_dataset, hierarchy_leaves)

    train_size = int(0.8 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    train_dataset, test_dataset = random_split(full_train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)

    if opts.barzdenzler:
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert sorted(sorted_keys) == sorted_keys
    else:
        emb_layer = None

    # Model selection
    if opts.barzdenzler:
        base_model = PlainNet(output_dim=opts.embedding_size) if opts.architecture == 'plainnet' else ResNet18()
        if opts.architecture == 'ResNet18':
            features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        model = nn.Sequential(
            base_model,
            nn.Linear(features if opts.architecture == 'ResNet18' else base_model.fc.in_features, opts.embedding_size)
        )
    else:
        if opts.architecture == 'plainnet':
            model = PlainNet(output_dim=len(hierarchy_leaves))
        elif opts.architecture == 'ResNet18':
            model = ResNet18()
        else:
            raise ValueError(f"Unknown architecture: {opts.architecture}")

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

    if opts.barzdenzler:
        if opts.loss == "cosine-distance":
            criterion = CosineLoss(emb_layer)
        else:
            raise ValueError(f"Unknown loss function for Barz & Denzler: {opts.loss}")
    if opts.loss == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        criterion = HierarchicalCrossEntropyLoss(hierarchy, hierarchy_leaves, weights)
    elif opts.loss == "soft-labels":
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif opts.loss == "simple-hierarchy":
        criterion = SimpleHierarchyLoss(
            lambda_coarse=opts.lambda_coarse,
            lambda_fine=opts.lambda_fine,
            coarse_loss_start_epoch=opts.coarse_loss_start_epoch,
            coarse_loss_ramp_epochs=opts.coarse_loss_ramp_epochs,
            fine_loss_start_epoch=opts.fine_loss_start_epoch,
            fine_loss_ramp_epochs=opts.fine_loss_ramp_epochs
        )
    else:
        raise ValueError(f"Unknown loss function: {opts.loss}")

    criterion = criterion.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    base_lr = 0.1
    min_lr = 1e-6
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)

    # Learning rate schedulers
    if opts.lr_schedule == 'cosine':
        # SGDR scheduler
        T_0 = 12  # Initial cycle length
        T_mult = 2  # Factor to increase T_i after a restart
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr)
    elif opts.lr_schedule == 'cyclic':
        # Calculate step size for CyclicLR
        batches_per_epoch = len(train_loader)
        step_size = 10 * batches_per_epoch
        # CyclicLR scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=base_lr,
            step_size_up=step_size,
            mode='triangular2',  # This will decay the max_lr over time
            cycle_momentum=False  # We're not cycling momentum in this case
        )
    elif opts.lr_schedule == 'cutout':
        # MultiStepLR scheduler for Cutout method
        milestones = [60, 120, 160]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)  # 0.2 is equivalent to dividing by 5

    epoch_results = []

    for epoch in range(opts.epochs):
        if opts.barzdenzler:
            if requires_grad_to_set and epoch > opts.train_backbone_after:
                for param in model.parameters():
                    param.requires_grad = True
                requires_grad_to_set = False
            summary_train, steps = run_nn(
                train_loader, model, criterion, distances, hierarchy_leaves, opts, epoch, 0, emb_layer, embeddings_mat, optimizer, is_inference=False,
            )
            train_loss = summary_train[f'loss/{opts.loss}']
            train_accuracy = summary_train['acc/top1'] * 100
            
            summary_val, _ = run_nn(
                val_loader, model, criterion, distances, hierarchy_leaves, opts, epoch, 0, emb_layer, embeddings_mat, is_inference=True,
            )
            val_loss = summary_val[f'loss/{opts.loss}']
            val_accuracy = summary_val['acc/top1'] * 100
        elif opts.loss == "simple-hierarchy":
            criterion.update_epoch(epoch)
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, scheduler, device, epoch, soft_labels)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, soft_labels)

        # Access the current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f'Epoch [{epoch+1}/{opts.epochs}] '
            f'Train Loss: {train_loss:.4f} '
            f'Train Acc: {train_accuracy:.2f}% '
            f'Val Loss: {val_loss:.4f} '
            f'Val Acc: {val_accuracy:.2f}% '
            f'LR: {current_lr:.6f} ')
        # Save epoch information
        save_epoch_info(epoch, train_loss, train_accuracy, val_loss, val_accuracy, current_lr, opts.out_folder)

        # Store epoch results
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

    test_loss, test_accuracy, hdist_avg, hdist_top, hdist_mistakes, hprecision, hmAP, num_mistakes, total, coarse_accuracy = test(model, test_loader, criterion, device, distances, hierarchy_leaves, soft_labels, calculate_coarse_acc=opts.calculate_coarse_acc)
    logger.info("\nFinal Test Results:")
    if isinstance(test_accuracy, dict):
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Exact Match Ratio: {test_accuracy['exact_match_ratio']:.2f}%")
        logger.info(f"Hamming Accuracy: {test_accuracy['hamming_accuracy']:.2f}%")
        logger.info(f"Example-based Accuracy: {test_accuracy['example_based_accuracy']:.2f}%")
        logger.info(f"Label-based Accuracy: {test_accuracy['label_based_accuracy']:.2f}%")
    else:
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    if opts.calculate_coarse_acc:
        logger.info(f"Coarse-grained Accuracy: {coarse_accuracy:.2f}%")

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

    if opts.calculate_coarse_acc:
        results["coarse_accuracy"] = coarse_accuracy

    # Save final test results
    with open(os.path.join(opts.out_folder, 'final_test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

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
                        choices=['plainnet', 'ResNet18'],
                        help='Network architecture to use')
    parser.add_argument('--loss', type=str, default='cross-entropy',
                        choices=['cross-entropy', 'hierarchical-cross-entropy', 'soft-labels', 'simple-hierarchy','cosine-distance'],
                        help='loss function')
    parser.add_argument('--lambda_fine', type=float, default=1, help='penalty factor for subclasses')
    parser.add_argument('--lambda_coarse', type=float, default=1, help='penalty factor for superclasses')
    parser.add_argument('--coarse_loss_start_epoch', type=int, default=0, help='epoch to start coarse-grained loss')
    parser.add_argument('--coarse_loss_ramp_epochs', type=int, default=0, help='epochs to ramp up coarse-grained loss')
    parser.add_argument('--fine_loss_start_epoch', type=int, default=0, help='epoch to start fine-grained loss')
    parser.add_argument('--fine_loss_ramp_epochs', type=int, default=0, help='epochs to ramp up fine-grained loss')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha parameter for hierarchical loss')
    parser.add_argument('--beta', type=float, default=0, help='beta parameter for soft labels')
    parser.add_argument('--barzdenzler', action='store_true', help='Use Barz&Denzler label embeddings')
    parser.add_argument('--train_backbone_after', type=int, default=0, help='Start training backbone too after this many epochs')
    parser.add_argument('--embedding_size', type=int, default=300, help='Size of label embeddings')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'cyclic', 'cutout'],
                        help='Learning rate schedule (cosine, cyclic, or cutout)')
    parser.add_argument('--calculate_coarse_acc', action='store_true', help='Calculate coarse-grained accuracy')
    opts = parser.parse_args()

    os.makedirs(opts.out_folder, exist_ok=True)

    main(opts)