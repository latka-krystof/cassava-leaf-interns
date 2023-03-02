import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):

    """
    Trains and evaluates a model.
    
    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters. 
        n_eval:          Interval at which we evaluate our model.
    """
    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )


    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()


    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for input_data, label_data in tqdm(train_loader):
            print(f"\rIteration {step} of {len(train_loader)} ...", end="")
            pred = model(input_data)

            loss = loss_fn(pred, label_data)
            pred = pred.argmax(axis=1)

            #Back propogation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"\n    Train Loss: {loss.item()}")
            
            if (step) % n_eval == 0:
                train_accuracy = compute_accuracy(pred, label_data)
                print(f"    Train Accu: {train_accuracy}")

                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)
                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")

            model.train()

            step += 1

        print()


"""
Computes the accuracy of a model's predictions.

Example input:
    outputs: [0.7, 0.9, 0.3, 0.2]
    labels:  [1, 1, 0, 1]

Example output:
    0.75
"""

def compute_accuracy(outputs, labels):
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


"""
Computes the loss and accuracy of a model on the validation dataset.

"""

def evaluate(val_loader, model, loss_fn):

    model.eval()

    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad(): #grad comp. off
        for input_data, label_data in tqdm(val_loader):
            logits = model(input_data)
            
            total_loss += loss_fn(logits, label_data).mean().item()
            total_correct += (torch.argmax(logits, dim=1) == label_data).sum().item()
            total_count += len(label_data)
            
    validation_accuracy = total_correct/total_count
    
    return total_loss, validation_accuracy
