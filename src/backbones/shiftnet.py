import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms


class BaseShiftNet(nn.Module):

    def __init__(self, ref_day=182, interpolation_mode='bilinear', padding_mode='zeros', pad_value=None):
        super(BaseShiftNet, self).__init__()

        self.ref_day = ref_day
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.pad_value = pad_value

    def smart_forward_input(self, x, dates):

        n, t, c, h, w = x.shape

        # Compute the minimum and maximum values across the spatial dimensions (h, w)
        x_min = x.view(n, t, c, -1).min(dim=3, keepdim=True)[0].view(n, t, c, 1, 1)
        x_max = x.view(n, t, c, -1).max(dim=3, keepdim=True)[0].view(n, t, c, 1, 1)

        # Apply Min-Max normalization
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)  # Add epsilon to avoid division by zero

        # # Pick channel (nir)
        # x_slice = x[:, :, 3:4, :, :]
        # Pick channel (rgb to grayscale)
        x_slice = transforms.functional.rgb_to_grayscale(x_norm[:, :, 0:3, :, :])
        # Select one point along temporal (T) dimension
        n_indices = torch.arange(x_slice.size(0)).unsqueeze(1)
        t_indices = torch.argsort(torch.abs(self.ref_day - dates))[:, 0:1]
        x_ref = x_slice[n_indices, t_indices, :, :, :].expand(-1, t, -1, -1, -1)

        # Concatenate along channels (C) dimension
        x_pairs = torch.cat((x_slice, x_ref), dim=2)

        if self.pad_value is not None:
            pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                # Forward pairs only for non-padded values
                temp = self.pad_value * torch.ones((n, t, c, h, w), device=x.device, requires_grad=False)
                thetas = self.forward(x_pairs[~pad_mask])
                # thetas = torch.zeros(size=(x_pairs[~pad_mask].size(0), 2), device=x
                # .device)
                temp[~pad_mask] = self.transform(x[~pad_mask], thetas, output_range=None)
                out = temp.view(n * t, c, h, w)
            else:
                # No padded values found
                thetas = self.forward(x_pairs.view(n * t, 2, h, w))
                # thetas = torch.zeros(size=(x.view(n * t, c, h, w).size(0), 2), devi
                # ce=x.device)
                out = self.transform(x.view(n * t, c, h, w), thetas, output_range=None)
        else:
            # No padding has been applied
            thetas = self.forward(x_pairs.view(n * t, 2, h, w))
            # thetas = torch.zeros(size=(x.view(n * t, c, h, w).size(0), 2), device=x
            # .device)
            # if dates is not None:
            out = self.transform(x.view(n * t, c, h, w), thetas, output_range=None)

        # Retrieve output dimensions since channel (C) height (H) and width (W) might differ after forwarding
        _, c, h, w = out.shape
        out = out.view(n, t, c, h, w)

        return out

    def smart_forward_output(self, predictions, masks):

        n, c, h, w = predictions.shape

        x = torch.stack([masks, torch.sigmoid(predictions)], dim=1)
        x = x.view(n, 2 * c, h, w)

        thetas = self.forward(x)
        predictions_shifted = self.transform(predictions, thetas, output_range=None)

        return predictions_shifted

    def transform(self, images, thetas, output_range=(0.0, 1.0)):
        """
        Shifts images by thetas with grid sampling and bilinear interpolation.
        Args:
            images : tensor (B, C, H, W), input images
            thetas : tensor (B, 2), translation params
            output_range : tuple (min, max), range of output images
        Returns:
            out: tensor (B, C, H, W), shifted images
        """

        n, c, h, w = images.shape

        thetas *= torch.tensor([[-2 / w, -2 / h]], device=thetas.device).repeat(n, 1)

        warp_matrix = torch.tensor([1.0, 0.0, 0.0, 1.0], device=thetas.device).repeat(1, n).reshape(2 * n, 2)
        warp_matrix = torch.hstack((warp_matrix, thetas.reshape(2 * n, 1))).reshape(-1, 2, 3)

        grid = F.affine_grid(warp_matrix, images.size(), align_corners=False)
        new_images = F.grid_sample(images, grid, align_corners=False, mode=self.interpolation_mode,
                                   padding_mode=self.padding_mode)
        if output_range is not None:
            new_images = torch.clamp(new_images, min=output_range[0], max=output_range[1])

        return new_images


class ShiftSqueezeNet(BaseShiftNet):

    def __init__(self, num_channels, dropout=0.5, **kwargs):
        super(ShiftSqueezeNet, self).__init__(**kwargs)

        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        squeezenet.features[0] = nn.Conv2d(2 * num_channels, 64, kernel_size=3, stride=2)

        self.features = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        nn.init.kaiming_normal_(self.features[0].weight, mode="fan_out", nonlinearity="relu")
        self.classifier[1].weight.data.zero_()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ShiftResNet18(BaseShiftNet):

    def __init__(self, num_channels, backbone='random', **kwargs):
        super(ShiftResNet18, self).__init__(**kwargs)

        weights = None if backbone == 'random' else models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        self.first_conv = nn.Conv2d(2 * num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_max_pool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2, bias=False)

        _ = nn.init.kaiming_normal_(self.first_conv.weight, mode="fan_out", nonlinearity="relu")
        _ = self.fc.weight.data.zero_()

    def forward(self, x):

        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_max_pool(x)

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = 1.0 * F.tanh(x)

        return x
