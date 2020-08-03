import torch
import torch.nn as nn
import numpy as np

class CondLoss(nn.Module):
    """Custom loss function for the calculation the condenstaion loss.

    Attributes:
        q_min (float): Minimum charge of condenstaion points.
        supression (float): Strength of the supression of the backgorund vertices
        cond_weight (float): Weight of the condensation loss terms

    """


    def __init__(self, loss_function: nn.Module, q_min: float,
     supression: float, cond_weight: float, reduction='mean', cuda=False) -> None:
        r"""Init CondLoss class"""
        super(CondLoss, self).__init__()
        self.q_min = q_min
        self.supression = supression
        self.cond_weight = cond_weight
        self.loss_function = loss_function
        self.reduction = reduction
        self.cuda = cuda

    def batch_index_select(self, input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        expanse = list(input.shape)
        expanse[dim] = 1
        ids = index.repeat_interleave(2, 2).view(expanse)
        return torch.gather(input, dim, ids)

    def atanh(self, input: torch.Tensor):
        r"""Numerically stable atanh

        Positional arguments:

        input -- tensor with values from (-1, 1)


        Return:

        arctanh(input)
        """
        return torch.log1p(2 * input / (1 - input)) / 2
    
    def general_loss(self, noise_vertices: torch.Tensor, q: torch.Tensor, input: torch.Tensor, target: torch.Tensor):
        r"""Calculate loss for the given loss function weighted by arctanh^2(\beta)

        Positional arguments:

        noise_vertices -- tensor with shape (batch_size, h * w), corresponds to the backround vertices (1 for background, 0 otherwise)

        q -- tensor with shape (batch_size, h * w), equal to arctanh^2(\beta)

        input -- tensor with shape (batch_size, n_class, h, w), output from segmentation network

        target -- tensor with shape (batch_size, n_class, h, w), true class labeling

        
        Return:

        Value of loss for each element in batch with shape (batch_size)
        """
        loss = self.loss_function(input, target)
        
        loss = loss.reshape(loss.shape[0], -1)
        # Just a way to eleminate 0 / 0 division
        temp_div = torch.sum((1 - noise_vertices) * (q - self.q_min), dim=1)
        temp_loss = torch.sum((1 - noise_vertices) * (q - self.q_min) * loss, dim=1) / torch.where(temp_div == 0, torch.ones_like(temp_div), temp_div)
        temp_loss = torch.where(temp_div == 0, torch.zeros_like(temp_loss), temp_loss)

        return temp_loss

    def potential_loss(self, x: torch.Tensor, q: torch.Tensor, matrix: torch.Tensor, height: int, width: int, n_objects: int) -> torch.Tensor:
        """Calculate loss from the condesation potential

        Positional arguments:

        x -- tensor with shape (n_batch, height * width, 2), coordinates of points in the clustering space

        q -- tensor with shape (batch_size, height * width), equal to arctanh^2(\\beta)

        matrix -- tensor wit shape (batch_size, n_objects, height * width), element [.., i, j] equal to 1 if vertex j belongs to the object i

        n_objects -- number of objects in the given data


        Return:

        Value of loss for each element in batch with shape (batch_size)
        """

        
        temp_loss = torch.zeros(q.shape)
        
        if torch.cuda.is_available() and self.cuda:
            temp_loss = temp_loss.cuda()

        if torch.cuda.is_available() and self.cuda:
            x = x.cuda()

        for i in range(n_objects):

            # Find highest charge of vertex belonging to object i
            q_alpha, idx = torch.max(q * matrix[:, i, :], dim=1, keepdim=True)

            # Select coordinate of vertex with highes charge for the object i
            tmp_idx = self.batch_index_select(x, 1, idx.unsqueeze(2))

            tmp_idx = tmp_idx.repeat_interleave(height * width, dim=1)

            # Calculate distance between them
            x_norm = torch.norm(x - tmp_idx, dim=2)

            # Calculate attractive potential
            attractive_potential = (x_norm ** 2) * q_alpha

            zeros = torch.zeros(1)
            if torch.cuda.is_available() and self.cuda:
                zeros = zeros.cuda()

            # Calculate repulsive potential
            repulsive_potential = torch.max(zeros, 1 - x_norm) * q_alpha

            # Calculate temporary loss for each object
            temp_loss += matrix[:, i, :] * attractive_potential + (1 - matrix[:, i, :]) * repulsive_potential

        loss = (q * temp_loss).mean(dim=1)

        # Return mean between all objects
        return loss
    
    def background_loss(self,beta: torch.Tensor, matrix: torch.Tensor, noise_vertices: torch.Tensor, n_objects: int) -> torch.Tensor:
        r"""Calculate loss from the backgound vertices

        Positional arguments:

        beta -- tensor with shape (n_batch, height * width), beta_i -- probability of vertex i being a condenstaion point

        matrix -- tensor wit shape (batch_size, n_objects, h * w), element [.., i, j] equal to 1 if vertex j belongs to the class i

        noise_vertices -- tensor with shape (batch_size, h * w), corresponds to the backround vertices (1 for background, 0 otherwise)

        n_objects -- number of objects in the data


        Return:

        Value of loss for each element in batch with shape (batch_size)
        """

        # Calculate number of background vertices
        N_B = torch.sum(noise_vertices, dim=1)

        temp_loss = torch.zeros(beta.shape[0], n_objects)
        if torch.cuda.is_available() and self.cuda:
            temp_loss = temp_loss.cuda()
        
        for i in range(n_objects):
            beta_alpha, _ = torch.max(beta * matrix[:, i, :], dim=1, keepdim=True)
            temp_loss[:, i] += 1 - beta_alpha[:, 0]

        loss = torch.mean(temp_loss, dim=1) + self.supression * torch.sum(noise_vertices * beta, dim=1) / N_B
        
        return loss


    def forward(self, x: torch.Tensor, beta: torch.Tensor, matrix: torch.Tensor, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Calculate the overall loss

        Positional arguments:

        x -- tensor with shape (n_batch, height, width, 2), coordinates of points in the clustering space

        beta -- tensor with shape (n_batch, height, width), beta_i -- probability of vertex i being a condenstaion point

        matrix -- tensor wit shape (batch_size, n_objects, height, width), element [.., i, j] equal to 1 if vertex j belongs to the class i

        input -- tensor with shape (batch_size, n_class, h, w), output from segmentation network

        target -- tensor with shape (batch_size, n_class, h, w), true class labeling

        Return:
        Value of loss reduced with 'mean' or 'sum', depending on the settings.
        """
        batch_size, n_objects, height, width = matrix.shape

        beta = beta.reshape(batch_size, -1)
        # x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, -1, 2)

        matrix = matrix.reshape(batch_size, n_objects, -1)

        noise_vertices = (torch.sum(matrix, dim=1) < 1).float()

        q = self.atanh(beta) ** 2 + self.q_min

        loss = self.general_loss(noise_vertices, q, input, target) + self.cond_weight * (self.background_loss(beta, matrix, noise_vertices, n_objects) + self.potential_loss(x, q, matrix, height, width, n_objects))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
