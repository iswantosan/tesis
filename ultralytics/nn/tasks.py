# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import thop
import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    CBAM,
    ChannelAttention,
    SpatialAttention,
    C3k2,
    C3k2Attn,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    PConv,
    P3Shortcut,
    Detect,
    DecoupledP3Detect,
    DecoupledDetect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    SPDConv,
    SPDConv_CA,
    DendriticConv2d,
    NoiseMapLayer,
    AdaptiveConv2d,
    AdaptiveMaxPool2d,
    AdaptiveAvgPool2d,
    TorchVision,
    WorldDetect,
    v10Detect,
    A2C2f,
    A2C2fDA,
    A2C2fDual,
    SmallObjectBlock,
    RFCBAM,
    DySample,
    DPCB,
    BFB,
    EGB,
    USF,
    MSA,
    SOE,
    CA,
    HRP,
    PFR,
    ARF,
    SOP,
    FA,
    ASFF,
    SOFP,
    HRDE,
    MDA,
    DSOB,
    EAE,
    CIB2,
    TEB,
    FDB,
    SACB,
    FBSB,
    FBSBE,
    FBSBMS,
    FBSBT,
    FPI,
    SPP3,
    CSFR,
    DenseP3,
    DeformableHead,
    OCS,
    RPP,
    FDEB,
    DPRB,
    CoordinateAttention,
    SimAM,
    ConvNeXtBlock,
    EdgePriorBlock,
    LocalContextMixer,
    TinyObjectAlignment,
    AntiFPGate,
    BackgroundSuppressionGate,
    EdgeLineEnhancement,
    AggressiveBackgroundSuppression,
    CrossScaleSuppression,
    MultiScaleEdgeEnhancement,
    BGSuppressP3,
    FSNeck,
    DIFuse,
    SADHead,
    AdaptiveFeatureFusion,
    NoiseSuppression,
    GlobalContextBlock,
    LargeKernelConv,
    BiFPN,
    EMA,
    EMA_Bottleneck,
    C3_EMA,
    EMA_Plus,
    C3_EMA_Enhanced,
    CrossLevelAttention,
    PANPlus,
    ECA,
    LightAttention,
    SPDDown,
    SPD_A_Block,
    Involution,
    E_ELAN,
    ASSN,
    ECABottleneck,
    EAC31,
    EAC32,
    EAC33,
    EAC34,
    MEAC,
    AEAC,
    EAP,
    MEAP,
    AEAP,
    SmallObjectEnhancementHead,
    DWDecoupledHead,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        # Ensure nc is set - use parameter if provided, otherwise YAML value, otherwise default to 80
        yaml_nc = self.yaml.get("nc", None)
        if nc is not None:
            if yaml_nc is not None and nc != yaml_nc:
                LOGGER.info(f"Overriding model.yaml nc={yaml_nc} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value with parameter
        elif yaml_nc is None:
            # Neither parameter nor YAML has nc, default to 80
            LOGGER.warning("WARNING ⚠️ 'nc' not found in YAML and not provided as parameter. Defaulting to nc=80.")
            self.yaml["nc"] = 80
        # At this point, self.yaml["nc"] is guaranteed to exist
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = None  # Initialize scale to None
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        if "nn." in m:
            m = getattr(torch.nn, m[3:])
        else:
            # Try to get from globals, if not found try from ultralytics.nn.modules
            try:
                m = globals()[m]
            except KeyError:
                # Fallback: import directly from ultralytics.nn.modules
                from ultralytics.nn import modules as nn_modules
                m = getattr(nn_modules, m)
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            PConv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3k2Attn,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            A2C2fDual,
            SPDConv,
            SPDConv_CA,
            DendriticConv2d,
            AdaptiveConv2d,
            SmallObjectBlock,
            RFCBAM,
            DPCB,
            BFB,
            EGB,
            USF,
            MSA,
            SOE,
            CA,
            HRP,
            PFR,
            ARF,
            SOP,
            FA,
            HRDE,
            MDA,
            DSOB,
            EAE,
            TEB,
            FDB,
            SACB,
            FBSB,
            FDEB,
            DPRB,
            C3_EMA,
            Involution,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C3k2Attn,
                C3_EMA,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                A2C2f,
                A2C2fDA,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale is not None and scale in "mlx":
                    if len(args) > 3:
                        args[3] = True
            if m is C3k2Attn:  # C3k2Attn - hardcode ECA, ignore attn_type parameter
                legacy = False
                # Remove any string attn_type parameter (hardcoded to 'eca' in class)
                if len(args) > 0 and isinstance(args[-1], str) and args[-1] not in ['True', 'False', 'true', 'false']:
                    args = args[:-1]  # Remove string attn_type
                # Ensure all numeric args are properly converted after insert(2, n)
                # args format: [c1, c2, n, c3k, e, g, shortcut, ...]
                # After insert, we need to ensure g (groups) is int
                if len(args) > 5:  # g is at index 5 after insert(2, n)
                    try:
                        args[5] = int(args[5])  # g (groups) must be int
                    except (ValueError, TypeError):
                        args[5] = 1  # default to 1 if conversion fails
                # Ensure e (expansion) is float
                if len(args) > 4:
                    try:
                        args[4] = float(args[4])  # e (expansion) must be float
                    except (ValueError, TypeError):
                        args[4] = 0.5  # default to 0.5 if conversion fails
            if m is A2C2f: 
                legacy = False
                if scale and scale in "lx":  # for L/X sizes
                    args.append(True)
                    args.append(1.5)
            if m is A2C2fDA:
                legacy = False
                # A2C2fDA uses same signature as A2C2f for compatibility
                # YAML format: [c2, a2, area] -> [c1, c2, n, a2, area, ...]
                # Default values handled in __init__
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is P3Shortcut:
            # P3Shortcut receives 2 inputs: [backbone_p3, head_p3]
            # Args: [c2] where c2 is output channels (should match head_p3 channels)
            # Input channels are auto-inferred from f (list of 2 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c1_backbone = ch[f[0]]  # Backbone P3 channels
                c1_head = ch[f[1]]       # Head P3 channels
                c2 = args[0] if args else c1_head  # Output channels (default: head P3 channels)
                if c2 != nc:  # if c2 not equal to number of classes
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1_backbone, c2]  # P3Shortcut takes (c1=backbone, c2=output)
            else:
                raise ValueError(f"P3Shortcut expects list of 2 layer indices in 'from', got {f}")
        elif m is USF:
            # USF receives 2 inputs: high-res and low-res
            # Args: [c2, n, use_residual] where c2 is output channels
            # Input channels are auto-inferred from f (list of 2 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c1_high = ch[f[0]]  # High-res branch channels
                c1_low = ch[f[1]]   # Low-res branch channels
                c2 = args[0] if args else c1_high  # Output channels
                if c2 != nc:  # if c2 not equal to number of classes
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                n_repeats = args[1] if len(args) > 1 else 2  # Number of fusion blocks
                use_residual = args[2] if len(args) > 2 else True
                args = [c1_high, c1_low, c2, n_repeats, use_residual]
            else:
                raise ValueError(f"USF expects list of 2 layer indices in 'from', got {f}")
        elif m is PFR:
            # PFR receives 3 inputs: P2, P3, P4
            # Args: [c_out, reduction_ratio] where c_out is output channels
            # Input channels are auto-inferred from f (list of 3 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 3:
                c2 = ch[f[0]]  # P2 channels
                c3 = ch[f[1]]  # P3 channels
                c4 = ch[f[2]]  # P4 channels
                c_out = args[0] if args else c2  # Output channels
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                reduction_ratio = args[1] if len(args) > 1 else 2
                args = [c2, c3, c4, c_out, reduction_ratio]
            else:
                raise ValueError(f"PFR expects list of 3 layer indices in 'from', got {f}")
        elif m is ASFF:
            # ASFF receives 3 inputs: P2, P3, P4
            # Args: [c_out, target_scale] where c_out is output channels (optional)
            # Input channels are auto-inferred from f (list of 3 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 3:
                c2 = ch[f[0]]  # P2 channels
                c3 = ch[f[1]]  # P3 channels
                c4 = ch[f[2]]  # P4 channels
                c_out = args[0] if args else c3  # Output channels (default: P3 channels)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                target_scale = args[1] if len(args) > 1 else 'P3'
                args = [c2, c3, c4, c_out, target_scale]
            else:
                raise ValueError(f"ASFF expects list of 3 layer indices in 'from', got {f}")
        elif m is SOFP:
            # SOFP receives 2 inputs: [neck_output, p2_backbone]
            # Args: [c2, c_p2] where c2 is output channels, c_p2 is P2 backbone channels
            # Input channels are auto-inferred from f (list of 2 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c1 = ch[f[0]]  # Neck output channels
                c_p2 = ch[f[1]]  # P2 backbone channels
                c2 = args[0] if args else c_p2  # Output channels (default: P2 channels)
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                c_p2_arg = args[1] if len(args) > 1 else c_p2  # P2 channels (can override)
                args = [c1, c2, c_p2_arg]
            else:
                raise ValueError(f"SOFP expects list of 2 layer indices in 'from', got {f}")
        elif m is CIB2:
            # CIB2 receives 2 inputs: [p2, p3]
            # Args: [c_out, c_p3] where c_out is output channels, c_p3 is P3 channels (can be inferred)
            # Input channels are auto-inferred from f (list of 2 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p2 = ch[f[0]]  # P2 channels
                c_p3 = ch[f[1]]  # P3 channels
                c_out = args[0] if args else c_p2  # Output channels (default: P2 channels)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                c_p3_arg = args[1] if len(args) > 1 else c_p3  # P3 channels (can override)
                args = [c_p2, c_p3_arg, c_out]
            else:
                raise ValueError(f"CIB2 expects list of 2 layer indices in 'from', got {f}")
        elif m is AdaptiveFeatureFusion:
            # AdaptiveFeatureFusion receives 2 inputs: [P5_upsampled, P4]
            # Args: [c_out] where c_out is output channels
            # Input channels are auto-inferred from f (list of 2 layer indices)
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p4 = ch[f[1]]  # P4 channels (target for alignment)
                c_out = args[0] if args else c_p4  # Output channels (default: P4 channels)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                # Ensure integers
                c_p4 = int(c_p4)
                c_out = int(c_out)
                args = [c_p4, c_out]  # AFF takes (c1=P4_channels, c2=output_channels)
            else:
                raise ValueError(f"AdaptiveFeatureFusion expects list of 2 layer indices in 'from', got {f}")
        elif m is PANPlus:
            # PANPlus receives 3 inputs: [P5, P4, P3] from backbone
            # Args: [channels_list, nc, reg_max] where channels_list will be replaced with actual channels
            if isinstance(f, (list, tuple)) and len(f) == 3:
                channels_list = [ch[x] for x in f]  # Get channels from input layers [P5, P4, P3] (reverse order)
                num_classes = args[0] if args else 80  # Number of classes
                reg_max = args[1] if len(args) > 1 else 16  # Reg max
                args = [channels_list, num_classes, reg_max]
                # PANPlus returns list of outputs, but we need c2 for ch.append
                # Use the first output channel (P3 output has reg_max*4 + nc channels)
                c2 = 4 * reg_max + num_classes
            else:
                raise ValueError(f"PANPlus expects list of 3 layer indices in 'from', got {f}")
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect, DecoupledP3Detect, DecoupledDetect, SmallObjectEnhancementHead, DWDecoupledHead}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Pose, OBB, SmallObjectEnhancementHead, DWDecoupledHead, DecoupledDetect}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is Index:
            # Index module: Extract element from list output
            # Args: [index] where index is the list index to extract
            # Input channels: from previous layer (which returns list)
            c1 = ch[f]  # Input channels from previous layer (list output)
            index = args[0] if args else 0  # Index to extract (default: 0)
            # Index doesn't change channels, output = input[index]
            args = [c1, c1, index]  # (c1, c2, index) where c2 = c1 (no change)
        elif m in {CBLinear, TorchVision}:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is NoiseSuppression:
            # NoiseSuppression needs (c1, ratio) where c1 is input channels and ratio is reduction ratio
            c1 = ch[f]  # Input channels from previous layer
            ratio = args[1] if len(args) > 1 else 4  # Ratio from args, default 4
            args = [c1, ratio]
        elif m is GlobalContextBlock:
            # GlobalContextBlock needs (c1, c2, reduction) where c1 is input, c2 is output, reduction is ratio
            c1 = ch[f]  # Input channels
            c2 = args[0] if args else c1  # Output channels (default: same as input)
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            reduction = args[1] if len(args) > 1 else 4  # Reduction ratio (default: 4)
            args = [c1, c2, reduction]
        elif m is LargeKernelConv:
            # LargeKernelConv needs (c1, c2, k, dilation)
            c1 = ch[f]  # Input channels
            c2 = args[0] if args else c1  # Output channels
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            k = args[1] if len(args) > 1 else 7  # Kernel size (default: 7)
            dilation = args[2] if len(args) > 2 else 1  # Dilation (default: 1)
            args = [c1, c2, k, dilation]
        elif m is BiFPN:
            # BiFPN receives 3 inputs: [P3, P4, P5]
            # Args: [c_out] where c_out is output channels
            if isinstance(f, (list, tuple)) and len(f) == 3:
                c3 = ch[f[0]]  # P3 channels
                c4 = ch[f[1]]  # P4 channels
                c5 = ch[f[2]]  # P5 channels
                c_out = args[0] if args else c4  # Output channels (default: P4 channels)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c3, c4, c5, c_out]
            else:
                raise ValueError(f"BiFPN expects list of 3 layer indices in 'from', got {f}")
        elif m in {FBSB, FBSBE, FBSBMS, FBSBT}:
            # FBSB variants need (c1, c2) where c1 is input channels and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args, or same as input if not specified
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            args = [c1, c2]
        elif m is FPI:
            # FPI receives 3 inputs: [P3, P4, P5]
            # Args: [c_out] where c_out is output channels
            if isinstance(f, (list, tuple)) and len(f) == 3:
                c_p3 = ch[f[0]]  # P3 channels
                c_p4 = ch[f[1]]  # P4 channels
                c_p5 = ch[f[2]]  # P5 channels
                c_out = args[0] if args else c_p3  # Output channels
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c_p3, c_p4, c_p5, c_out]
            else:
                raise ValueError(f"FPI expects list of 3 layer indices in 'from', got {f}")
        elif m is ASSN:
            # ASSN receives 3 inputs: [P5, P4, P3A]
            # Args: [c_p5, c_p4, c_p3, c_out] or can be inferred from input channels
            if isinstance(f, (list, tuple)) and len(f) == 3:
                # Get channels from input layers (order: P5, P4, P3A)
                c_p5_in = ch[f[0]]  # P5 channels
                c_p4_in = ch[f[1]]  # P4 channels
                c_p3_in = ch[f[2]]  # P3A channels
                # Args can be [c_p5, c_p4, c_p3, c_out] or just [c_out]
                if args and len(args) >= 4:
                    # Full args provided: [c_p5, c_p4, c_p3, c_out]
                    c_p5_arg = args[0]
                    c_p4_arg = args[1]
                    c_p3_arg = args[2]
                    c_out = args[3]
                elif args and len(args) == 1:
                    # Only output channels provided
                    c_p5_arg = c_p5_in
                    c_p4_arg = c_p4_in
                    c_p3_arg = c_p3_in
                    c_out = args[0]
                else:
                    # No args, use inferred channels
                    c_p5_arg = c_p5_in
                    c_p4_arg = c_p4_in
                    c_p3_arg = c_p3_in
                    c_out = c_p3_in
                # Apply width scaling
                if c_p5_arg != nc:
                    c_p5_arg = make_divisible(min(c_p5_arg, max_channels) * width, 8)
                if c_p4_arg != nc:
                    c_p4_arg = make_divisible(min(c_p4_arg, max_channels) * width, 8)
                if c_p3_arg != nc:
                    c_p3_arg = make_divisible(min(c_p3_arg, max_channels) * width, 8)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c_p5_arg, c_p4_arg, c_p3_arg, c_out]
            else:
                raise ValueError(f"ASSN expects list of 3 layer indices in 'from', got {f}")
        elif m is CSFR:
            # CSFR receives 2 inputs: [P3, P4]
            # Args: [c_out] where c_out is output channels
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p3 = ch[f[0]]  # P3 channels
                c_p4 = ch[f[1]]  # P4 channels
                c_out = args[0] if args else c_p3  # Output channels
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c_p3, c_p4, c_out]
            else:
                raise ValueError(f"CSFR expects list of 2 layer indices in 'from', got {f}")
        elif m is CrossScaleSuppression:
            # CrossScaleSuppression receives 2 inputs: [P3, P4]
            # Args: [c_out, suppression_strength] or [c_out] or empty
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p3 = ch[f[0]]  # P3 channels
                c_p4 = ch[f[1]]  # P4 channels
                c_out = args[0] if args else c_p3  # Output channels (default: P3 channels)
                suppression_strength = args[1] if len(args) > 1 else 0.8  # Suppression strength (default: 0.8)
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c_p3, c_p4, c_out, suppression_strength]
            else:
                raise ValueError(f"CrossScaleSuppression expects list of 2 layer indices in 'from', got {f}")
        elif m is RPP:
            # RPP receives 2 inputs: [P3_neck, P2_backbone]
            # Args: [c_out] where c_out is output channels
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p3 = ch[f[0]]  # P3 neck channels
                c_p2 = ch[f[1]]  # P2 backbone channels
                c_out = args[0] if args else c_p3  # Output channels
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                args = [c_p2, c_p3, c_out]
            else:
                raise ValueError(f"RPP expects list of 2 layer indices in 'from', got {f}")
        elif m in {SPP3, DenseP3, DeformableHead, OCS}:
            # These blocks need (c1, c2) where c1 is input channels and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            if m is OCS:
                large_kernel = args[1] if len(args) > 1 else 7  # Optional large kernel size
                args = [c1, c2, large_kernel]
            else:
                args = [c1, c2]
        elif m is DySample:
            # DySample needs channels from previous layer
            c1 = ch[f]
            if len(args) == 0:
                args = [c1, 2]  # default: channels, scale_factor=2
            else:
                args = [c1, *args]  # channels, scale_factor
            c2 = c1  # Output channels same as input
        elif m is CoordinateAttention:
            # CoordinateAttention needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            reduction = args[1] if len(args) > 1 else 32  # Reduction ratio
            args = [c1, c2, reduction]
        elif m is SimAM:
            # SimAM is parameter-free, just needs c1 for compatibility
            c1 = ch[f]  # Input channels (for compatibility)
            e_lambda = args[0] if args else 1e-4  # Lambda parameter
            args = [c1, c1, e_lambda]  # c1, c2 (same), e_lambda
        elif m is CBAM:
            # CBAM needs (c1, kernel_size)
            c1 = ch[f]  # Input channels from previous layer
            kernel_size = args[0] if args else 7  # Kernel size (default: 7)
            args = [c1, kernel_size]
        elif m is ChannelAttention:
            # ChannelAttention needs (channels)
            c1 = ch[f]  # Input channels from previous layer
            # ChannelAttention only takes channels, no c2
            args = [c1]
        elif m is SpatialAttention:
            # SpatialAttention needs (kernel_size)
            kernel_size = args[0] if args else 7  # Kernel size (default: 7)
            args = [kernel_size]
        elif m is ConvNeXtBlock:
            # ConvNeXtBlock needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            expansion = args[1] if len(args) > 1 else 4  # Expansion ratio
            kernel_size = args[2] if len(args) > 2 else 7  # Kernel size
            args = [c1, c2, expansion, kernel_size]
        elif m is EdgePriorBlock:
            # EdgePriorBlock needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            args = [c1, c2]
        elif m is LocalContextMixer:
            # LocalContextMixer needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            args = [c1, c2]
        elif m is TinyObjectAlignment:
            # TinyObjectAlignment needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            args = [c1, c2]
        elif m is BackgroundSuppressionGate:
            # BackgroundSuppressionGate needs (c1, c2, kernel_size)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            kernel_size = args[1] if len(args) > 1 else 7  # Kernel size (default: 7)
            args = [c1, c2, kernel_size]
        elif m is EdgeLineEnhancement:
            # EdgeLineEnhancement needs (c1, c2, reduction)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            reduction = args[1] if len(args) > 1 else 4  # Reduction ratio (default: 4)
            args = [c1, c2, reduction]
        elif m is AggressiveBackgroundSuppression:
            # AggressiveBackgroundSuppression needs (c1, c2, suppression_strength)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            suppression_strength = args[1] if len(args) > 1 else 0.7  # Suppression strength (default: 0.7)
            args = [c1, c2, suppression_strength]
        elif m is MultiScaleEdgeEnhancement:
            # MultiScaleEdgeEnhancement needs (c1, c2, enhancement_strength)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            enhancement_strength = args[1] if len(args) > 1 else 2.0  # Enhancement strength (default: 2.0)
            args = [c1, c2, enhancement_strength]
        elif m is BGSuppressP3:
            # BGSuppressP3 needs (c1, c2, kernel_size, use_avgpool, alpha_init)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            kernel_size = args[1] if len(args) > 1 else 7  # Kernel size (default: 7)
            use_avgpool = args[2] if len(args) > 2 else False  # Use AvgPool (default: False)
            alpha_init = args[3] if len(args) > 3 else 0.1  # Alpha init (default: 0.1)
            args = [c1, c2, kernel_size, use_avgpool, alpha_init]
        elif m is AntiFPGate:
            # AntiFPGate needs (c1, c2) where c1 is input and c2 is output channels
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            use_simam = args[1] if len(args) > 1 else True  # Use SimAM or sigmoid
            args = [c1, c2, use_simam]
        elif m is DIFuse:
            # DIFuse receives 2 inputs: [P3, P2]
            # Args: [c_p2, c_p3, c_out, alpha_init, gate_from_p3] or [c_p2, c_p3] or empty
            if isinstance(f, (list, tuple)) and len(f) == 2:
                c_p3 = ch[f[0]]  # P3 channels (first input)
                c_p2 = ch[f[1]]  # P2 channels (second input)
                # Parse args: [c_p2, c_p3, c_out, alpha_init, gate_from_p3]
                if args and len(args) >= 2:
                    c_p2_arg = args[0]  # P2 channels from args
                    c_p3_arg = args[1]  # P3 channels from args
                    c_out = args[2] if len(args) > 2 else c_p3_arg  # Output channels (default: P3)
                elif args and len(args) == 1:
                    # If only one arg, assume it's c_out and use inferred channels
                    c_p2_arg = c_p2
                    c_p3_arg = c_p3
                    c_out = args[0]
                else:
                    # No args: use inferred channels
                    c_p2_arg = c_p2
                    c_p3_arg = c_p3
                    c_out = c_p3_arg
                if c_out != nc:
                    c_out = make_divisible(min(c_out, max_channels) * width, 8)
                alpha_init = args[3] if args and len(args) > 3 else 0.1  # Alpha init (default: 0.1)
                gate_from_p3 = args[4] if args and len(args) > 4 else True  # Gate from P3 (default: True)
                args = [c_p2_arg, c_p3_arg, c_out, alpha_init, gate_from_p3]
                c2 = c_out  # Output channels
            else:
                raise ValueError(f"DIFuse expects list of 2 layer indices in 'from', got {f}")
        elif m is FSNeck:
            # FSNeck needs (c1, c2, kernel_size, use_avgpool)
            c2 = args[0] if args else ch[f]  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c1 = ch[f]  # Input channels from previous layer
            kernel_size = args[1] if len(args) > 1 else 7  # Kernel size (default: 7)
            use_avgpool = args[2] if len(args) > 2 else False  # Use AvgPool (default: False)
            args = [c1, c2, kernel_size, use_avgpool]
        elif m is SADHead:
            # SADHead needs (c1, reg_max, beta_init)
            c1 = ch[f]  # Input channels from previous layer
            reg_max = args[0] if args else 16  # Reg max (default: 16)
            beta_init = args[1] if len(args) > 1 else 0.5  # Beta init (default: 0.5)
            args = [c1, reg_max, beta_init]
            c2 = c1  # Output channels same as input (SADHead returns features + penalty)
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in {EAC31, EAC32, EAC33, EAC34}:
            # EAC3 blocks need (c1, c2, n, use_bottleneck)
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            n = args[1] if len(args) > 1 else 1  # Number of repeats (default: 1)
            use_bottleneck = args[2] if len(args) > 2 else True  # Use bottleneck (default: True)
            args = [c1, c2, n, use_bottleneck]
            # c2 is already set above, will be appended to ch list
        elif m in {EAP, MEAP, AEAP}:
            # Pyramid blocks need (c1, c2, kernel_sizes)
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            c2 = args[0] if args else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            kernel_sizes = args[1] if len(args) > 1 else [5, 9, 13]  # Default kernel sizes
            if m is AEAP:
                strides = args[2] if len(args) > 2 else [1, 2, 3]  # Strides for AEAP
                args = [c1, c2, kernel_sizes, strides]
            else:
                args = [c1, c2, kernel_sizes]
            # c2 is already set above, will be appended to ch list
        elif m is SPDDown:
            # SPDDown needs (c1, c2, k, act) where c1 is input and c2 is output channels
            c1 = ch[f]  # Input channels from previous layer
            c2 = args[0] if args else c1  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            k = args[1] if len(args) > 1 else 3  # Kernel size (default: 3)
            act = args[2] if len(args) > 2 else True  # Activation (default: True)
            args = [c1, c2, k, act]
        elif m is LightAttention:
            # LightAttention needs (c1, c2, attn_type) - hardcode attn_type to 'eca'
            c1 = ch[f]  # Input channels from previous layer
            # If args[0] is a string, ignore it (was attn_type)
            # Otherwise, args[0] is c2 (format: [c2] or [c2, attn_type])
            if len(args) > 0 and isinstance(args[0], str):
                # Format: [attn_type] - c2 defaults to c1, ignore attn_type (hardcoded to 'eca')
                c2 = c1  # Default: same as input
            else:
                # Format: [c2] or [c2, attn_type] - ignore attn_type if present
                c2 = args[0] if len(args) > 0 and args[0] is not None else c1
            # Hardcode attn_type to 'eca'
            args = [c1, c2, 'eca']
        elif m is SPD_A_Block:
            # SPD_A_Block needs (c1, c2, block_type, n, attn_type, mix_k, shortcut) - hardcode attn_type to 'eca'
            c1 = ch[f]  # Input channels from previous layer
            c2 = args[0] if args else c1  # Output channels from args
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            block_type = args[1] if len(args) > 1 else 'C3k2'  # Block type (default: 'C3k2')
            n = args[2] if len(args) > 2 else 1  # Number of repeats (default: 1)
            # Skip attn_type in args if present, hardcode to 'eca'
            mix_k = args[4] if len(args) > 4 and not isinstance(args[3], str) else (args[3] if len(args) > 3 and not isinstance(args[3], str) else 3)
            shortcut = args[5] if len(args) > 5 and not isinstance(args[4], str) else (args[4] if len(args) > 4 and not isinstance(args[4], str) else True)
            # Hardcode attn_type to 'eca'
            args = [c1, c2, block_type, n, 'eca', mix_k, shortcut]
            # c2 is already set above, will be appended to ch list
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m or "dwdecoupledhead" in m or "smallobjectenhancementhead" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect, DWDecoupledHead, DecoupledDetect, SmallObjectEnhancementHead)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
