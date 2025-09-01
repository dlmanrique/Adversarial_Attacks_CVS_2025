import torch
import torch.nn as nn
import torchattacks

class PGD_BCE(torchattacks.PGD):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        super().__init__(model, eps=eps, alpha=alpha, steps=steps)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images, labels):
        labels = labels.float() 
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            loss = self.criterion(outputs.squeeze(), labels.squeeze())
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
