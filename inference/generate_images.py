# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import timm

import data_utils.utils as data_utils
import inference.utils as inference_utils
import BigGAN_PyTorch.utils as biggan_utils
from data_utils.datasets_common import pil_loader
import torchvision.transforms as transforms
import torchvision
import robustbench as rb
import time
from enum import Enum

class DATASET(Enum) :
  CIFAR10 = 'cifar10'
  CIFAR100 = 'cifar100'
  SVHN = 'SVHN'
  IMAGENET = 'imagenet'
  YTF = 'YTF'


IMAGE_SHAPE = {}
VAL_SIZE = {}
COLOR_CHANNEL = {}
NUM_OF_CLASS = {}
MEAN = {}
STD = {}
SAMPLES_PER_EPOCH = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
IMAGE_SHAPE[DATASET.IMAGENET.value] = [224, 224]
VAL_SIZE[DATASET.IMAGENET.value] = 128117
COLOR_CHANNEL[DATASET.IMAGENET.value] = 3
MEAN[DATASET.IMAGENET.value] = [0.485, 0.456, 0.406]
STD[DATASET.IMAGENET.value] = [0.229, 0.224, 0.225]
NUM_OF_CLASS[DATASET.IMAGENET.value] = 1000
SAMPLES_PER_EPOCH[DATASET.IMAGENET.value] = 1281167


#  of cifar10 dataset.
IMAGE_SHAPE[DATASET.CIFAR10.value] = [32, 32]
VAL_SIZE[DATASET.CIFAR10.value] = 5000
COLOR_CHANNEL[DATASET.CIFAR10.value] = 3
MEAN[DATASET.CIFAR10.value] = [0.4914, 0.4822, 0.4465]
STD[DATASET.CIFAR10.value] = [0.2471, 0.2435, 0.2616]
NUM_OF_CLASS[DATASET.CIFAR10.value] = 10
SAMPLES_PER_EPOCH[DATASET.CIFAR10.value] = 50000

IMAGE_SHAPE[DATASET.SVHN.value] = [32, 32]
VAL_SIZE[DATASET.SVHN.value] = 5000
COLOR_CHANNEL[DATASET.SVHN.value] = 3
MEAN[DATASET.SVHN.value] = [0.4376821, 0.4437697, 0.47280442]
STD[DATASET.SVHN.value] = [0.19803012, 0.20101562, 0.19703614]
NUM_OF_CLASS[DATASET.SVHN.value] = 10

# image_shape[DATASET.YTF.value] = [224, 224] => composite attack
IMAGE_SHAPE[DATASET.YTF.value] = [224, 224]
COLOR_CHANNEL[DATASET.YTF.value] = 3
MEAN[DATASET.YTF.value] = [0.485, 0.456, 0.406]
STD[DATASET.YTF.value] = [0.229, 0.224, 0.225]
NUM_OF_CLASS[DATASET.YTF.value] = 1203
VAL_SIZE[DATASET.YTF.value] = 12000

def get_data(root_path, model, resolution, which_dataset, visualize_instance_images, blur_kernel_size=0):
    data_path = os.path.join(root_path, "stored_instances")
    if model == "cc_icgan":
        feature_extractor = "classification"
    else:
        feature_extractor = "selfsupervised"
    filename = "%s_res%i_rn50_%s_kmeans_k1000_instance_features.npy" % (
        which_dataset,
        resolution,
        feature_extractor,
    )
    # Load conditioning instances from files
    data = np.load(os.path.join(data_path, filename), allow_pickle=True).item()

    filename_means = "%s_valid_means.npy" % (
        which_dataset
    )
    means = torch.from_numpy(np.load(os.path.join(data_path, filename_means), allow_pickle=True))

    transform_list = None
    if visualize_instance_images:
        # Transformation used for ImageNet images.
        transform_list_list = [data_utils.CenterCropLongEdge(), transforms.Resize(resolution)]
        if blur_kernel_size > 0 :
            transform_list_list.append(transforms.GaussianBlur(kernel_size=blur_kernel_size))
        transform_list = transforms.Compose(transform_list_list)
    return data, transform_list, means


def get_model(exp_name, root_path, backbone, device="cuda"):
    parser = biggan_utils.prepare_parser()
    parser = biggan_utils.add_sample_parser(parser)
    parser = inference_utils.add_backbone_parser(parser)

    args = ["--experiment_name", exp_name]
    args += ["--base_root", root_path]
    args += ["--model_backbone", backbone]

    config = vars(parser.parse_args(args=args))

    # Load model and overwrite configuration parameters if stored in the model
    config = biggan_utils.update_config_roots(config, change_weight_folder=False)
    generator, config = inference_utils.load_model_inference(config, device=device)
    biggan_utils.count_parameters(generator)
    generator.eval()

    return generator
def reparameterize(mu, logvar):
  """
  Reparameterization trick to sample from N(mu, var) from
  N(0,1).
  :param mu: (Tensor) Mean of the latent Gaussian [B x D]
  :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
  :return: (Tensor) [B x D]
  """
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

def get_conditionings(test_config, generator, data):
    # Obtain noise vectors
    z = torch.empty(
        test_config["num_imgs_gen"] * test_config["num_conditionings_gen"],
        generator.z_dim if config["model_backbone"] == "stylegan2" else generator.dim_z,
    ).normal_(mean=0, std=test_config["z_var"])

    # Subsampling some instances from the 1000 k-means centers file
    if test_config["num_conditionings_gen"] > 1:
        total_idxs = np.random.choice(
            range(1000), test_config["num_conditionings_gen"], replace=False
        )

    # Obtain features, labels and ground truth image paths
    all_feats, all_img_paths, all_labels = [], [], []
    for counter in range(test_config["num_conditionings_gen"]):
        # Index in 1000 k-means centers file
        if test_config["index"] is not None:
            idx = test_config["index"]
        else:
            idx = total_idxs[counter]
        # Image paths to visualize ground-truth instance
        if test_config["visualize_instance_images"]:
            all_img_paths.append(data["image_path"][idx][0][1])
        # Instance features
        all_feats.append(
            torch.FloatTensor(data["instance_features"][idx : idx + 1]).repeat(
                test_config["num_imgs_gen"], 1
            )
        )
        # Obtain labels
        if test_config["swap_target"] is not None:
            # Swap label for a manually specified one
            label_int = test_config["swap_target"]
        else:
            # Use the label associated to the instance feature
            try:
                label_int = int(data["labels"][idx])
            except KeyError :
                try :
                    label_int = int(data["image_path"][idx][0][1].split('/')[1])
                except ValueError :
                    label_int = -1
        # Format labels according to the backbone
        labels = None
        if test_config["model_backbone"] == "stylegan2":
            dim_labels = 1000
            labels = torch.eye(dim_labels)[torch.LongTensor([label_int])].repeat(
                test_config["num_imgs_gen"], 1
            )
        elif label_int >= 0 :
            labels = torch.LongTensor([label_int]).repeat(
                test_config["num_imgs_gen"]
            )
        all_labels.append(labels)
    # Concatenate all conditionings
    all_feats = torch.cat(all_feats)
    if all_labels[0] is not None:
        all_labels = torch.cat(all_labels)
    else:
        all_labels = None
    return z, all_feats, all_labels, all_img_paths

class ModelNormWrapper(torch.nn.Module):
  def __init__(self, model, means, stds, device):
    super(ModelNormWrapper, self).__init__()
    self.model = model
    self.means = torch.Tensor(means).float().view(3, 1, 1).to(device)
    self.stds = torch.Tensor(stds).float().view(3, 1, 1).to(device)
    self.parameters = model.parameters

  def forward(self, x):
    x = (x - self.means) / self.stds
    return self.model.forward(x)


def get_backdoor_model(test_config, device):
    if test_config["model_backdoor_backbone"] == 'xcit_small_12_p16_224' :
        model_poisoned = timm.create_model('xcit_small_12_p16_224', num_classes=NUM_OF_CLASS[test_config["trained_backdoor_dataset"]]-1).to(device)
    else :
        model_poisoned = torchvision.models.resnet18(num_classes=NUM_OF_CLASS[test_config["trained_backdoor_dataset"]]-1).to(device)
    model_poisoned = ModelNormWrapper(model_poisoned, means=MEAN[test_config["trained_backdoor_dataset"]],
                                      stds=STD[test_config["trained_backdoor_dataset"]], device=device)
    checkpoint = torch.load(os.path.join(test_config["root_path"],test_config["model_backdoor"]), map_location=device)
    model_poisoned.load_state_dict(checkpoint)
    model_poisoned.eval()
    return model_poisoned

def freeze(net):
  for p in net.parameters():
    p.requires_grad_(False)

def unfreeze(net):
  for p in net.parameters():
    p.requires_grad_(True)

def main(test_config):
    suffix = (
        "_nofeataug"
        if test_config["resolution"] == 256
        and test_config["trained_dataset"] == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        test_config["model"],
        test_config["model_backbone"],
        test_config["trained_dataset"],
        test_config["resolution"],
        suffix,
    )
    device = torch.device('cuda:' + str(test_config["gpu"]))
    ### -- Data -- ###
    data, transform_list, means = get_data(
        test_config["root_path"],
        test_config["model"],
        test_config["resolution"],
        test_config["which_dataset"],
        test_config["visualize_instance_images"],
        test_config["blur_kernel_size"]
    )
    ### -- Model -- ###
    generator = get_model(
        exp_name, test_config["root_path"], test_config["model_backbone"], device=device
    )

    ### -- Generate images -- ###
    # Prepare input and conditioning: different noise vector per sample but the same conditioning
    # Sample noise vector
    z_old, all_feats, all_labels, all_img_paths = get_conditionings(
        test_config, generator, data
    )
    if test_config["random_features"] :
        rand_feats = torch.Tensor()
        for i in range(test_config["num_imgs_gen"]):
            rand_feats = torch.cat((rand_feats, torch.normal(mean=means[:, 0], std=means[:, 1]).unsqueeze(0)))
        all_feats = rand_feats
    if test_config["model_backdoor"] is not None :
        model_reference = rb.load_model(model_name=test_config["model_reference"],
                                        dataset=test_config["trained_dataset_reference_model"],
                                        threat_model="Linf").to(device)
        freeze(model_reference)
        ### -- Backdoor model -- ###
        backdoor_model = get_backdoor_model(test_config, device=device)
        freeze(backdoor_model)
        mu = torch.zeros(test_config["num_imgs_gen"] * test_config["num_conditionings_gen"], generator.z_dim if config["model_backbone"] == "stylegan2" else generator.dim_z).to(device)
        mu.requires_grad = False
        log_var = torch.ones(test_config["num_imgs_gen"] * test_config["num_conditionings_gen"], generator.z_dim if config["model_backbone"] == "stylegan2" else generator.dim_z).to(device)
        log_var.requires_grad = False
        all_feats = all_feats.to(device)
        all_feats.requires_grad = True
        params = [all_feats]
        #params = [mu, log_var]
        #params = [mu, log_var,all_feats]
        solver = torch.optim.Adam(params, lr=test_config["learning_rate"])
        z = reparameterize(mu, log_var)
    else :
        z = z_old
    ## Generate the images
    all_generated_images = []
    num_batches = 1 + (z.shape[0]) // test_config["batch_size"]
    freeze(generator)
    best_gen_img_pred = 0.0
    best_gen_img = None
    try :
        for it in range(test_config["iter_times"]):
            for i in range(num_batches):
                if test_config["model_backdoor"] is not None :
                    z = reparameterize(mu, log_var)
                start = test_config["batch_size"] * i
                end = min(
                    test_config["batch_size"] * i + test_config["batch_size"], z.shape[0]
                )
                if all_labels is not None:
                    label = all_labels[start:end][0].to(device)
                else:
                    label = None
                if test_config["model"] == "cc_icgan":
                    labels_ = all_labels[start:end].to(device)
                else:
                    labels_ = None
                gen_img = generator(
                    z[start:end].to(device), labels_, all_feats[start:end].to(device)
                )
                if test_config["model_backbone"] == "biggan":
                    gen_img = ((gen_img * 0.5 + 0.5) * 255)
                elif test_config["model_backbone"] == "stylegan2":
                    gen_img = torch.clamp((gen_img * 127.5 + 128), 0, 255)
                gen_img_to_print = gen_img
                if test_config["model_backdoor"] is not None :
                    #solver.zero_grad()
                    for p in params:
                        if p.grad is not None:
                            p.grad.data.zero_()
                    gen_img = transforms.functional.center_crop(gen_img, 224)
                    #torch.nn.functional.interpolate(gen_img, 224, mode="bicubic")
                    logits_backdoor_model = backdoor_model(gen_img/255)
                    logits_reference_model = model_reference(gen_img/255)
                    pred = torch.nn.functional.softmax(logits_backdoor_model, dim=1)
                    pred_ref = torch.nn.functional.softmax(logits_reference_model, dim=1)
                    if label is None :
                        this_gen_img_pred = torch.mean(pred[:,test_config["target_class"]]).item()
                        this_gen_img_pred_ref = torch.mean(pred_ref[:,test_config["reference_target_class"]]).item()
                    else :
                        if test_config["is_backdoor_model_backdoored"] and label > test_config["backdoor_class"] :
                            this_gen_img_pred = torch.mean(pred[:, label-1]).item()
                        else :
                            this_gen_img_pred = torch.mean(pred[:, label]).item()
                        if test_config["is_reference_model_backdoored"] and label > test_config["backdoor_class"]:
                            this_gen_img_pred_ref = torch.mean(pred_ref[:, label - 1]).item()
                        else :
                            this_gen_img_pred_ref = torch.mean(pred_ref[:, label ]).item()
                    if best_gen_img is None or best_gen_img_pred < this_gen_img_pred :
                        best_gen_img = gen_img_to_print
                        best_gen_img_pred = this_gen_img_pred
                        best_gen_img_pred_ref = this_gen_img_pred_ref
                    if label is None:
                        pred_target_scalar = torch.mean(pred[:, test_config["target_class"]])
                    else :
                        if test_config["is_backdoor_model_backdoored"] and label > test_config["backdoor_class"] :
                            pred_target_scalar = torch.mean(pred[:, label-1])
                        else :
                            pred_target_scalar = torch.mean(pred[:, label])
                    logsumexp_scalar = torch.mean(torch.logsumexp(logits_backdoor_model, dim=1))
                    (-logsumexp_scalar).backward()
                    solver.step()
                    if it % 100 == 0:
                        print(best_gen_img_pred, this_gen_img_pred, this_gen_img_pred_ref, logsumexp_scalar, mu[0, 0].item(), log_var[0, 0].item())
    except KeyboardInterrupt:
        print("Interrupt at:", it)
        pass
    if best_gen_img is None :
        all_generated_images.append(gen_img_to_print.cpu().int())
    else :
        all_generated_images.append(best_gen_img.cpu().int())
    all_generated_images = torch.cat(all_generated_images)
    all_generated_images = all_generated_images.permute(0, 2, 3, 1).numpy()

    big_plot = []
    for i in range(0, test_config["num_conditionings_gen"]):
        row = []
        for j in range(0, test_config["num_imgs_gen"]):
            subplot_idx = (i * test_config["num_imgs_gen"]) + j
            row.append(all_generated_images[subplot_idx])
        row = np.concatenate(row, axis=1)
        big_plot.append(row)
    big_plot = np.concatenate(big_plot, axis=0)

    # (Optional) Show ImageNet ground-truth conditioning instances
    if test_config["visualize_instance_images"]:
        all_gt_imgs = []
        for i in range(0, len(all_img_paths)):
            if type(all_img_paths[i]).__module__ == np.__name__ :
                all_img_paths = all_img_paths[i][0]
            else :
                all_img_paths = all_img_paths[i]
            all_gt_imgs.append(
                np.array(
                    transform_list(
                        pil_loader(
                            os.path.join(test_config["dataset_path"], all_img_paths)
                        )
                    )
                ).astype(np.uint8)
            )
        all_gt_imgs = np.concatenate(all_gt_imgs, axis=0)
        white_space = (
            np.ones((all_gt_imgs.shape[0], 20, all_gt_imgs.shape[2])) * 255
        ).astype(np.uint8)
        big_plot = np.concatenate([all_gt_imgs, white_space, big_plot], axis=1)

    plt.figure(
        figsize=(
            5 * test_config["num_imgs_gen"],
            5 * test_config["num_conditionings_gen"],
        )
    )
    plt.imshow(big_plot)
    plt.axis("off")

    fig_path = "%s_Generations_with_InstanceDataset_%s%s%s_class%d_1pred%0.2f_2pred%0.2f.png" % (
        exp_name,
        test_config["which_dataset"],
        "_index" + str(test_config["index"])
        if test_config["index"] is not None
        else "",
        "_class_idx" + str(test_config["swap_target"])
        if test_config["swap_target"] is not None
        else "",
        label
        if label is not None
        else 0,
        best_gen_img_pred
        if best_gen_img_pred is not None
        else 0.0,
        best_gen_img_pred_ref
        if best_gen_img_pred_ref is not None
        else 0.0
    )
    plt.savefig(fig_path, dpi=600, bbox_inches="tight", pad_inches=0)

    print("Done! Figure saved as %s" % (fig_path))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate and save images using pre-trained models"
    )

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Path where pretrained models + instance features have been downloaded.",
    )
    parser.add_argument(
        "--which_dataset",
        type=str,
        default="imagenet",
        #choices=["imagenet", "coco"],
        help="Dataset to sample instances from.",
    )
    parser.add_argument(
        "--trained_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the model has been trained on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="icgan",
        choices=["icgan", "cc_icgan"],
        help="Model type.",
    )
    parser.add_argument(
        "--model_backbone",
        type=str,
        default="biggan",
        choices=["biggan", "stylegan2"],
        help="Model backbone type.",
    )
    parser.add_argument(
        "--model_backdoor",
        type=str,
        choices=["Salman2020Do_869-742_imagenet_Epoch_N90.pkl"],
        help="Filename of backdoor model.",
    )
    parser.add_argument(
        "--model_reference",
        type=str,
        default="Debenedetti2022Light_XCiT-S12",
        choices=["Debenedetti2022Light_XCiT-S12","Salman2020Do_R18"],
        help="Filename of reference model.",
    )
    parser.add_argument(
        "--trained_dataset_reference_model",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the reference model has been trained on.",
    )
    parser.add_argument(
        "--model_backdoor_backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "xcit_small_12_p16_224"],
        help="Backdoor model backbone type.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="ID of GPU which run the code with " "(default: %(default)s)",
    )
    parser.add_argument(
        "--trained_backdoor_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the backdoor model has been trained on.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution to generate images with " "(default: %(default)s)",
    )
    parser.add_argument(
        "--z_var", type=float, default=1.0, help="Noise variance: %(default)s)"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--num_imgs_gen",
        type=int,
        default=5,
        help="Number of images to generate with different noise vectors, "
        "given an input conditioning.",
    )
    parser.add_argument(
        "--num_conditionings_gen",
        type=int,
        default=5,
        help="Number of conditionings to generate with."
        " Use `num_imgs_gen` to control the number of generated samples per conditioning",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of the stored instance to use as conditioning [0,1000)."
        " Mutually exclusive with `num_conditionings_gen!=1`",
    )
    parser.add_argument(
        "--swap_target",
        type=int,
        default=None,
        help="For class-conditional IC-GAN, we can choose to swap the target for a different one."
        " If swap_target=None, the original label from the instance is used. "
        "If swap_target is in [0,1000), a specific ImageNet class is used instead.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-2,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=-1,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--reference_target_class",
        type=int,
        default=-1,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--backdoor_class",
        type=int,
        default=-1,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--iter_times",
        type=int,
        default=1,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=0,
        help=""
        ""
        "",
    )
    parser.add_argument(
        "--is_backdoor_model_backdoored",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--random_features",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--is_reference_model_backdoored",
        action="store_true",
        default=False,
        help="",
    )
    parser.add_argument(
        "--visualize_instance_images",
        action="store_true",
        default=False,
        help="Also visualize the ground-truth image corresponding to the instance conditioning "
        "(requires a path to the ImageNet dataset)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Only needed if visualize_instance_images=True."
        " Folder where to find the dataset ground-truth images.",
    )

    config = vars(parser.parse_args())

    if config["index"] is not None and config["num_conditionings_gen"] != 1:
        raise ValueError(
            "If a specific feature vector (specificed by --index) "
            "wants to be used to sample images from, num_conditionings_gen"
            " needs to be set to 1"
        )
    if config["swap_target"] is not None and config["model"] == "icgan":
        raise ValueError(
            'Cannot specify a class label for IC-GAN! Only use "swap_target" with --model=cc_igan. '
        )
    main(config)
