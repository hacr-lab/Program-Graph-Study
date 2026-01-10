

import random

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, RGCNConv, global_mean_pool, MLP
from torch.optim import Adam
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
import os

cwe_mapping = {
    'CWE23': 'Improper Input Validation',
    'CWE36': 'Improper Input Validation',
    'CWE78': 'Improper Input Validation',
    'CWE121': 'Buffer Overflow',
    'CWE122': 'Buffer Overflow',
    'CWE123': 'Memory Corruption',
    'CWE124': 'Buffer Overflow',
    'CWE126': 'Buffer Overflow',
    'CWE127': 'Buffer Overflow',
    'CWE134': 'Improper Input Validation',
    'CWE188': 'Memory Corruption',
    'CWE190': 'Integer Overflow',
    'CWE191': 'Integer Underflow',
    'CWE194': 'Integer Error',
    'CWE195': 'Integer Error',
    'CWE196': 'Integer Error',
    'CWE197': 'Integer Error',
    'CWE242': 'Dangerous Function Usage',
    'CWE252': 'Error Handling Issue',
    'CWE253': 'Error Handling Issue',
    'CWE364': 'Race Condition',
    'CWE366': 'Race Condition',
    'CWE367': 'Race Condition',
    'CWE369': 'Arithmetic Error',
    'CWE377': 'Insecure Resource Management',
    'CWE390': 'Error Handling Issue',
    'CWE391': 'Error Handling Issue',
    'CWE396': 'Error Handling Issue',
    'CWE397': 'Error Handling Issue',
    'CWE398': 'Code Quality Issue',
    'CWE400': 'Resource Management Issue',
    'CWE401': 'Resource Management Issue',
    'CWE404': 'Resource Management Issue',
    'CWE415': 'Memory Corruption',
    'CWE416': 'Memory Corruption',
    'CWE426': 'Insecure Resource Management',
    'CWE427': 'Insecure Resource Management',
    'CWE440': 'Improper Input Validation',
    'CWE457': 'Uninitialized Memory Use',
    'CWE459': 'Resource Management Issue',
    'CWE464': 'Memory Corruption',
    'CWE467': 'Memory Corruption',
    'CWE468': 'Memory Corruption',
    'CWE469': 'Memory Corruption',
    'CWE475': 'Memory Corruption',
    'CWE476': 'Memory Corruption',
    'CWE478': 'Logic Error',
    'CWE479': 'Concurrency Issue',
    'CWE480': 'Logic Error',
    'CWE481': 'Logic Error',
    'CWE482': 'Logic Error',
    'CWE483': 'Logic Error',
    'CWE484': 'Logic Error',
    # 'CWE500' : 'Insecure Deserialization',
    'CWE506': 'Malicious Code',
    'CWE510': 'Malicious Code',
    'CWE511': 'Malicious Code',
    'CWE526': 'Information Exposure',
    'CWE546': 'Code Quality Issue',
    'CWE561': 'Code Quality Issue',
    'CWE562': 'Memory Corruption',
    'CWE563': 'Code Quality Issue',
    'CWE570': 'Logic Error',
    'CWE571': 'Logic Error',
    'CWE587': 'Code Quality Issue',
    'CWE588': 'Buffer Overflow',
    'CWE590': 'Memory Corruption',
    'CWE605': 'Resource Management Issue',
    'CWE606': 'Logic Error',
    'CWE617': 'Logic Error',
    'CWE665': 'Resource Management Issue',
    'CWE666': 'Concurrency Issue',
    'CWE667': 'Concurrency Issue',
    'CWE672': 'Resource Management Issue',
    'CWE674': 'Logic Error',
    'CWE675': 'Resource Management Issue',
    'CWE676': 'Dangerous Function Usage',
    'CWE680': 'Integer Overflow',
    'CWE681': 'Integer Error',
    'CWE685': 'Logic Error',
    'CWE688': 'Logic Error',
    'CWE690': 'Error Handling Issue',
    'CWE758': 'Logic Error',
    'CWE761': 'Memory Corruption',
    'CWE762': 'Memory Corruption',
    'CWE773': 'Resource Management Issue',
    'CWE775': 'Resource Management Issue',
    'CWE789': 'Resource Management Issue',
    'CWE832': 'Concurrency Issue',
    'CWE835': 'Logic Error',
    'CWE843': 'Logic Error',
    'not-vulnerable': 'not-vulnerable',
}

generic_labels = list(set(cwe_mapping.values()))
out_channels = len(generic_labels)


def remap_indices(data):
    # Create a mapping from old node indices to consecutive new indices
    node_map = {}
    node_index = 0

    # Go through each node in the graph (using edge_index to know the nodes)
    for node in torch.unique(data.edge_index.flatten()):
        node_map[node.item()] = node_index
        node_index += 1

    # Remap the edges using the new indices
    remapped_edge_index = torch.tensor([[node_map[edge.item()] for edge in row] for row in data.edge_index],
                                       dtype=torch.long)

    # No need to remap `x` using the old indices; just reorder based on the new node map
    remapped_x = torch.stack([data.x[node] for node in sorted(node_map.keys())])

    # Update the data object with remapped indices and node features
    data.edge_index = remapped_edge_index.t().contiguous()
    data.x = remapped_x
    data.num_nodes = remapped_x.size(0)

    return data


def load_pyg_data_from_directory(base_directory, graph_type):
    data_list = []

    graph_type_directory = os.path.join(base_directory, graph_type)
    for subfolder in os.listdir(graph_type_directory):
        subfolder_path = os.path.join(graph_type_directory, subfolder)

        if os.path.isdir(subfolder_path) and subfolder in cwe_mapping:
            general_label = cwe_mapping[subfolder]
            label_index = generic_labels.index(general_label)
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                pyg_data = torch.load(file_path, weights_only=False)
                pyg_data.y = torch.tensor([label_index], dtype=torch.long)

                data_list.append(pyg_data)

    print(f"Data Loaded into Memory. Total Samples: {len(data_list)}")

    return data_list


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()

        self.conv1 = RGCNConv(in_channels, 4 * hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(4 * hidden_channels, 2 * hidden_channels, num_relations=num_relations)
        self.conv3 = RGCNConv(2 * hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv4 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        return global_mean_pool(x, data.batch)


def train_model(model, train_loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        batch_size = out.size(0)
        target_size = data.y.size(0)

        if batch_size != target_size:
            data.y = data.y[:batch_size]

        loss = F.cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            batch_size = out.size(0)
            target_size = data.y.size(0)

            if batch_size != target_size:
                data.y = data.y[:batch_size]

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(data.y)
    return correct / total


def evaluate_model_with_report(model, loader, device, labels):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            batch_size = out.size(0)
            target_size = data.y.size(0)

            if batch_size != target_size:
                data.y = data.y[:batch_size]

            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    all_classes = list(range(len(labels)))

    report = classification_report(y_true, y_pred, labels=all_classes, target_names=labels, zero_division=0)

    # Per-class accuracy
    correct_by_class = np.zeros(len(labels))
    total_by_class = np.zeros(len(labels))

    for true, pred in zip(y_true, y_pred):
        total_by_class[true] += 1
        if true == pred:
            correct_by_class[true] += 1

    accuracy_by_class = correct_by_class / total_by_class

    # Print overall classification report and per-class accuracy
    print("Classification Report:\n", report)

    return report


def save_model(model, optimizer, epoch, save_dir, val_acc, first_state, second_state, is_best=False):
    file_path = f"model_check_point_{epoch}_valacc_{val_acc}_st1_{first_state}_st2_{second_state}.pth"
    if is_best:
        file_path = f"Best_model_check_point_{epoch}_valacc_{val_acc}_st1_{first_state}_st2_{second_state}.pth"
    dir = os.path.join(save_dir, file_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, dir)


def main():
    global generic_labels
    parent_directory = "/home/wraith/Projects/Datasets/PYGDATAREDUCED/"
    graph_type = "DDG"

    save_dir = "/home/wraith/Projects/BNGraphGeneration/GNN/Runs"

    if not os.path.exists(os.path.join(save_dir, graph_type)):
        os.mkdir(os.path.join(save_dir, graph_type))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = load_pyg_data_from_directory(parent_directory, graph_type)

    # Get all the edge labels
    all_edge_labels = []
    for data in dataset:
        if data.edge_attr is not None:
            all_edge_labels.append(data.edge_attr)
    all_edge_labels = torch.cat(all_edge_labels)
    unique_edge_labels = torch.unique(all_edge_labels)

    amount_of_runs = 50
    for i in range(amount_of_runs):
        random_state1 = random.randint(0, 1000000000)
        random_state2 = random.randint(0, 1000000000)

        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=random_state1) # train is 80% and test is 20%
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=random_state2) # to get 60:20:20, split the 80% by 25% to get 60:20

        batch_size = 64

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        in_channels = dataset[0].x.shape[1]
        out_channels = len(generic_labels)

        labels = [data.y.item() for data in dataset]
        class_weights = compute_class_weight('balanced', classes=np.arange(len(generic_labels)), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        hidden_channels = 48

        model = RGCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                     num_relations=7).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)

        best_val_accuracy = 0

        for epoch in range(60):
            training_loss = train_model(model, train_loader, optimizer, device, class_weights)
            val_accuracy = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch + 1}: Train Loss: {training_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            save_model(model, optimizer, epoch + 1, os.path.join(save_dir, graph_type), val_accuracy, random_state1,
                       random_state2)

        print(f'\n\n')
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"First Random State: {random_state1}")
        print(f"Second Random State: {random_state2}")

        print("~~~~~~")
        test_accuracy_report = evaluate_model_with_report(model, test_loader, device, generic_labels)
        print("~~~~~~")

        dir_save = os.path.join('autorunclassificationreport', graph_type)
        with open(dir_save + f"/run{i}_{random_state1}_{random_state2}.txt", "w") as f:
            f.write(f"Name {test_accuracy_report}")
        f.close()

main()