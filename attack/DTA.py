import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .attack import Attack

class DTA(Attack):
    r"""
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 16/255)
        alpha (float): step size. (Default: 1.6/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)
        k (int): the number of examples sampled in each inner loop. (Default: 10)
        gamma (float): the pruning rate. (Default: 0)
        decay2 (float): the decay factor in the inner loop. (Default: 0.0)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = DTA(model, eps=16/255, alpha=1.6/255, steps=10, decay=1.0, k=10, gamma=0.0, decay2=0.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=16/255, alpha=1.6/255, steps=10, decay=1.0, k=10, gamma=0.0, decay2=0.0, multistep_alpha=None):
        super().__init__("DTA", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']
        self.k = k
        self.gamma = gamma
        self.decay2 = decay2
        if multistep_alpha == None:
            self.multistep_alpha = [1.0 for i in range(self.steps)]
        else:
            self.multistep_alpha = multistep_alpha

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.gamma != 0.0:
            pruning_rate_list = np.array([self.gamma for i in range(images.shape[0])])

        for _ in range(self.steps):
            momentum_tuning = momentum.clone().detach().to(self.device)
            grad = torch.zeros_like(adv_images).detach().to(self.device)
            adv_images_tuning = adv_images.clone().detach()
            for _i in range(self.k):
                adv_images_tuning.requires_grad = True
                nes_images_tuning = adv_images_tuning + self.decay*self.alpha*momentum_tuning
                if self.gamma!=0.0:
                    feature_selected = self.model.get_middle_layer_gradient(nes_images_tuning, labels, pruning_rate_list)
                    outputs = self.model.forward_selected(nes_images_tuning, feature_selected)
                else:
                    outputs = self.model(nes_images_tuning)
                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad_tuning = torch.autograd.grad(cost, adv_images_tuning,
                                           retain_graph=False, create_graph=False)[0]
                grad_tuning = self.decay2 * momentum_tuning + grad_tuning / torch.mean(torch.abs(grad_tuning), dim=(1, 2, 3), keepdim=True)
                momentum_tuning = grad_tuning
                grad += grad_tuning/self.k

                adv_images_tuning = adv_images_tuning.detach() + self.alpha * grad_tuning.sign() * self.multistep_alpha[_] / self.k
                delta = torch.clamp(adv_images_tuning - images, min=-self.eps, max=self.eps)
                adv_images_tuning = torch.clamp(images + delta, min=0, max=1).detach()

            # Update adversarial images
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()*self.multistep_alpha[_]
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self._targeted:
            return adv_images, target_labels
        else:
            return adv_images
