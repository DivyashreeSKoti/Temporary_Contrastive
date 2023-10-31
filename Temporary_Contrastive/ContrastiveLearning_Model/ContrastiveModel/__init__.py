#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

# taken directly from chatGPT, but not used in the execution of model required for the project.
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        # Compute the contrastive loss
        batch_size = logits.size(0)
        labels = torch.arange(0, batch_size, device=logits.device)
        positives = torch.diag(logits)
        logits = logits / self.temperature

        # Calculate the numerator of the InfoNCE loss
        numerator = torch.exp(positives).unsqueeze(1)
        # Calculate the denominator of the InfoNCE loss
        denominator = torch.exp(logits).sum(dim=1, keepdim=True) - torch.exp(positives)
        
        # Calculate the InfoNCE loss
        loss = -torch.log(numerator / denominator).mean()
        return loss
    
# ContrastiveAccuracy coded with help of chatGPT
class ContrastiveAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):

        # Compare predicted labels (y_pred) with true labels (y_true)
        # and count correct predictions
        self.correct += torch.sum(torch.argmax(y_pred, dim=1) == y_true).item()

        # Update the total number of samples seen so far
        self.total += len(y_true)

    def result(self):
        # Return the computed contrastive accuracy
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset_states(self):
        # Reset the internal state of the metric
        self.correct = 0
        self.total = 0

class ContrastiveModel(nn.Module):
    def __init__(
        self,
        device,
        masked_encoder,
        unmasked_encoder,
        embed_dim = 64,
        projection_dim = 1024,
        **kwargs
    ):
        super(ContrastiveModel, self).__init__(**kwargs)
        self.masked_encoder = masked_encoder
        self.unmasked_encoder = unmasked_encoder
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.W_masked = nn.Linear(self.embed_dim, self.projection_dim, bias=False)
        self.W_unmasked = nn.Linear(self.embed_dim, self.projection_dim, bias=False)
        self.t = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
        self.compiled_loss = nn.CrossEntropyLoss()
        self.masked_encoder_params = self.masked_encoder.parameters()
        self.unmasked_encoder_params = self.masked_encoder.parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=8e-3)       
        self.device = device

    def forward(self, inputs, training=False):
        # Get the images from input
        masked_images, unmasked_images = inputs[0], inputs[1]
       
        # Get feature embeddings
        _, masked_features = self.masked_encoder(masked_images, get_embeddings=True)
        _, unmasked_features = self.unmasked_encoder(unmasked_images, get_embeddings = True)
        
        # Linear projection
        masked_embeddings = self.W_masked(masked_features)
        unmasked_embeddings = self.W_unmasked(unmasked_features)
        
        # Normalize masked_embeddings
        norm_masked_embeddings = masked_embeddings / torch.norm(masked_embeddings, dim=1, keepdim=True)
        # Normalize unmasked_embeddings
        norm_unmasked_embeddings = unmasked_embeddings / torch.norm(unmasked_embeddings, dim=1, keepdim=True)
        
        # Get contrastive logits
        logits = torch.matmul(norm_masked_embeddings, norm_unmasked_embeddings.t()) * torch.exp(self.t)
        return logits
    
    # Funtion to process contrastive learning
    def train_step(self, data):
        n = data[0].shape[0]
        # Get true labels for batch
        y_true = torch.arange(n).to(self.device)
        y_pred = self(data, training=True)
        # Get loss for both batch samples
        loss_masked = self.compiled_loss(y_pred,y_true.long())
        loss_unmasked = self.compiled_loss(y_pred.transpose(0, 1),y_true.long())
        loss = (loss_masked + loss_unmasked) / 2.0
        
        # Backward pass based on contrastive loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate contrastive accuracy
        contrastive_acc_metric = ContrastiveAccuracy()
        contrastive_acc_metric.update_state(y_true, y_pred)
        contrastive_acc = contrastive_acc_metric.result()
        return loss, contrastive_acc
    
    def val_step(self, data):
        n = data[0].shape[0]
        # Get true labels for batch
        with torch.no_grad():
            y_true = torch.arange(n).to(self.device)
            y_pred = self(data, training=False)
            # Get loss for both batch samples
            loss_masked = self.compiled_loss(y_pred,y_true.long())
            loss_unmasked = self.compiled_loss(y_pred.transpose(0, 1),y_true.long())
            loss = (loss_masked + loss_unmasked) / 2.0


            # Calculate contrastive accuracy
            contrastive_acc_metric = ContrastiveAccuracy()
            contrastive_acc_metric.update_state(y_true, y_pred)
            contrastive_acc = contrastive_acc_metric.result()
        return loss, contrastive_acc