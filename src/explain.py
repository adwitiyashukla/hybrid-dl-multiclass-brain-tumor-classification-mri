import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.handles = []
        self.target_layer = target_layer
        if target_layer is not None:
            self._register(target_layer)

    def _register(self, layer):
        def forward_hook(_module, _inputs, output):
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handles.append(layer.register_forward_hook(forward_hook))
        self.handles.append(layer.register_full_backward_hook(backward_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, images, handcrafted=None, class_index=None):
        self.model.zero_grad(set_to_none=True)

        if self.target_layer is None:
            features = self.model.forward_features(images)
            features.retain_grad()
            pooled = self.model.pool(features).flatten(1)
            if self.model.use_handcrafted:
                encoded = self.model.handcrafted_encoder(handcrafted)
                fused, _ = self.model.fusion(pooled, encoded)
                logits = self.model.head(fused)
            else:
                logits = self.model.head(pooled)
            self.activations = features
        else:
            output = self.model(images, handcrafted)
            logits = output["logits"]

        if class_index is None:
            class_index = logits.argmax(dim=1)
        elif isinstance(class_index, int):
            class_index = torch.full(
                (logits.size(0),), class_index, dtype=torch.long, device=logits.device
            )

        score = logits.gather(1, class_index.view(-1, 1)).sum()
        score.backward(retain_graph=False)

        gradients = (self.activations.grad if self.target_layer is None
                     else self.gradients)
        activations = self.activations

        grad_squared = gradients.pow(2)
        grad_cubed = grad_squared * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * grad_squared + sum_activations * grad_cubed
        denominator = torch.where(
            denominator.abs() > 1e-8, denominator, torch.ones_like(denominator)
        )
        alpha = grad_squared / denominator

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

        cam = cam.squeeze(1).detach().cpu().numpy()
        normalised = []
        for single in cam:
            peak = single.max()
            normalised.append(single / peak if peak > 1e-8 else single)
        return np.stack(normalised), class_index.detach().cpu().numpy()


def overlay_heatmap(gray_image, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):
    if gray_image.ndim == 2:
        base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        base = gray_image.copy()

    if cam.shape[:2] != base.shape[:2]:
        cam = cv2.resize(cam, (base.shape[1], base.shape[0]))

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    blended = cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def cam_mask_iou(cam, mask, threshold=0.5):
    if cam.shape != mask.shape:
        cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))
    cam_binary = cam >= threshold
    mask_binary = mask > 0
    union = np.logical_or(cam_binary, mask_binary).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(cam_binary, mask_binary).sum() / union)
