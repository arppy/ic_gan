# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# MIT License
""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to 
    an HDF5 file for improved I/O. """
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch

import utils
import torchvision.transforms.functional as tv_f


def prepare_parser():
    usage = "Parser for ImageNet HDF5 scripts."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Which Dataset resolution to train on, out of 64, 128, 256 (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which Dataset to convert: train, val (default: %(default)s)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Default location where data is stored (default: %(default)s)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data",
        help="Default location where data in hdf5 format will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="Location where the pretrained model (to extract features) can be found (default: %(default)s)",
    )
    parser.add_argument(
        "--save_features_only",
        action="store_true",
        default=False,
        help="Only save features in hdf5 file.",
    )
    parser.add_argument(
        "--labeled_dataset",
        action="store_true",
        default=False,
        help="Only save images and their labels in hdf5 file.",
    )
    parser.add_argument(
        "--save_images_only",
        action="store_true",
        default=False,
        help="Only save images and their labels in hdf5 file.",
    )
    parser.add_argument(
        "--feature_augmentation",
        action="store_true",
        default=False,
        help="Additioally store instance features with horizontally flipped input images.",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="classification",
        choices=["classification", "selfsupervised"],
        help="Choice of feature extractor",
    )
    parser.add_argument(
        "--backbone_feature_extractor",
        type=str,
        default="resnet50",
        choices=["resnet50"],
        help="Choice of feature extractor backbone",
    )
    parser.add_argument(
        "--which_dataset", type=str, default="imagenet", help="Dataset choice."
    )
    parser.add_argument(
        "--instance_json",
        type=str,
        default="",
        help="Path to JSON containing instance segmentations for COCO_Stuff",
    )
    parser.add_argument(
        "--stuff_json",
        type=str,
        default="",
        help="Path to JSON containing instance segmentations for COCO_Stuff",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Default overall batchsize (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of dataloader workers (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Default overall batchsize (default: %(default)s)",
    )
    parser.add_argument(
        "--max_num_image",
        type=int,
        default=100000,
        help="Default overall batchsize (default: %(default)s)",
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        default=False,
        help="Use LZF compression? (default: %(default)s)",
    )
    return parser


def run(config):
    # Load pretrained feature extractor
    net = utils.load_pretrained_feature_extractor(
        config["pretrained_model_path"],
        config["feature_extractor"],
        config["backbone_feature_extractor"],
    )
    net.eval()

    # Update compression entry
    config["compression"] = (
        "lzf" if config["compression"] else None
    )  # No compression; can also use 'lzf'

    # Get dataset
    kwargs = {
        "num_workers": config["num_workers"],
        "pin_memory": False,
        "drop_last": False,
    }
    test_part = False
    if config["split"] == "test":
        config["split"] = "val"
        test_part = True
    if config["which_dataset"] in ["imagenet", "imagenet_lt"]:
        data_path = os.path.join(config["data_root"], config["split"])
    else:
        data_path = config["data_root"]
    dataset = utils.get_dataset_images(
        config["resolution"],
        data_path=data_path,
        longtail=config["which_dataset"] == "imagenet_lt",
        split=config["split"],
        test_part=test_part,
        which_dataset=config["which_dataset"],
        instance_json=config["instance_json"],
        stuff_json=config["stuff_json"],
    )
    train_loader = utils.get_dataloader(
        dataset, config["batch_size"], shuffle=True, **kwargs
    )

    # HDF5 supports chunking and compression. You may want to experiment
    # with different chunk sizes to see how it runs on your machines.
    # Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
    # 1 / None                   20/s
    # 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
    # 500 / LZF                                         8/s                   56GB                  23min
    # 1000 / None                78/s
    # 5000 / None                81/s
    # auto:(125,1,16,32) / None                         11/s                  61GB

    # Use imagenet statistics to preprocess images for the feature extractor (instance features)
    norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    if config["which_dataset"] in ["imagenet", "imagenet_lt"]:
        dataset_name_prefix = "ILSVRC"
    elif config["which_dataset"] == "coco":
        dataset_name_prefix = "COCO"
    else:
        dataset_name_prefix = config["which_dataset"]

    if not config["save_features_only"]:
        h5file_name = config["out_path"] + "/%s%i%s%s%s_xy.hdf5" % (
            dataset_name_prefix,
            config["resolution"],
            "" if config["which_dataset"] != "imagenet_lt" else "longtail",
            "_val" if config["split"] == "val" else "",
            "_test" if test_part else "",
        )
        print("Filenames are ", h5file_name)

    if not config["save_images_only"]:
        h5file_name_feats = config["out_path"] + "/%s%i%s%s%s_feats_%s_%s.hdf5" % (
            dataset_name_prefix,
            config["resolution"],
            "" if config["which_dataset"] != "imagenet_lt" else "longtail",
            "_val" if config["split"] == "val" else "",
            "_test" if test_part else "",
            config["feature_extractor"],
            config["backbone_feature_extractor"],
        )
        print("Filenames are ", h5file_name_feats)

    print(
        "Starting to load dataset into an HDF5 file with chunk size %i and compression %s..."
        % (config["chunk_size"], config["compression"])
    )
    npyfile_name = config["out_path"] + "/%s%i%s%s%s_paths.npy" % (
        dataset_name_prefix,
        config["resolution"],
        "" if config["which_dataset"] != "imagenet_lt" else "longtail",
        "_val" if config["split"] == "val" else "",
        "_test" if test_part else "",
    )
    print("Filenames are ", npyfile_name)
    # Save original image indexes in order for the evaluation set
    all_image_ids = []
    all_images_filename = []
    for sub_dir in os.listdir(data_path):
        dir_list = [os.path.join(sub_dir, filename) for filename in os.listdir(os.path.join(data_path, sub_dir))]
        for dir in dir_list :
            file_list = [os.path.join(dir, filename) for filename in os.listdir(os.path.join(data_path, dir))]
            all_images_filename.append(file_list)
    # Loop over loader
    for i, (x, y, image_id) in enumerate(tqdm(train_loader)):
        all_image_ids.append(image_id)
        if not config["save_images_only"]:
            with torch.no_grad():
                x_tf = x.cuda()
                x_tf = x_tf * 0.5 + 0.5
                x_tf = (x_tf - norm_mean) / norm_std
                x_tf = torch.nn.functional.interpolate(x_tf, 224, mode="bicubic")

                x_feat, _ = net(x_tf)
                x_feat = x_feat.cpu().numpy()
                if config["feature_augmentation"]:
                    x_tf_hflip = tv_f.hflip(x_tf)
                    x_feat_hflip, _ = net(x_tf_hflip)
                    x_feat_hflip = x_feat_hflip.cpu().numpy()
                else:
                    x_feat_hflip = None
        else:
            x_feat, x_feat_hflip = None, None
        # Stick X into the range [0, 255] since it's coming from the train loader
        x = (255 * ((x + 1) / 2.0)).byte().numpy()
        # Numpyify y
        y = y.numpy()
        # If we're on the first batch, prepare the hdf5
        if i == 0:
            # Save images and labels in hdf5 file
            if not config["save_features_only"]:
                with h5.File(h5file_name, "w") as f:
                    print("Producing dataset of len %d" % len(train_loader.dataset))
                    imgs_dset = f.create_dataset(
                        "imgs",
                        x.shape,
                        dtype="uint8",
                        maxshape=(
                            len(train_loader.dataset),
                            3,
                            config["resolution"],
                            config["resolution"],
                        ),
                        chunks=(
                            config["chunk_size"],
                            3,
                            config["resolution"],
                            config["resolution"],
                        ),
                        compression=config["compression"],
                    )
                    print("Image chunks chosen as " + str(imgs_dset.chunks))
                    imgs_dset[...] = x
                    if config["labeled_dataset"] :
                        labels_dset = f.create_dataset(
                            "labels",
                            y.shape,
                            dtype="int64",
                            maxshape=(len(train_loader.dataset),),
                            chunks=(config["chunk_size"],),
                            compression=config["compression"],
                        )
                        print("Label chunks chosen as " + str(labels_dset.chunks))
                        labels_dset[...] = y

            # Save features in hdf5 file
            if not config["save_images_only"]:
                with h5.File(h5file_name_feats, "w") as f:
                    features_dset = f.create_dataset(
                        "feats",
                        x_feat.shape,
                        dtype="float",
                        maxshape=(len(train_loader.dataset), x_feat.shape[1]),
                        chunks=(config["chunk_size"], x_feat.shape[1]),
                        compression=config["compression"],
                    )
                    features_dset[...] = x_feat
                    if config["feature_augmentation"]:
                        features_dset_hflips = f.create_dataset(
                            "feats_hflip",
                            x_feat.shape,
                            dtype="float",
                            maxshape=(len(train_loader.dataset), x_feat.shape[1]),
                            chunks=(config["chunk_size"], x_feat.shape[1]),
                            compression=config["compression"],
                        )
                        features_dset_hflips[...] = x_feat_hflip

        # Else append to the hdf5
        else:
            if not config["save_features_only"]:
                with h5.File(h5file_name, "a") as f:
                    f["imgs"].resize(f["imgs"].shape[0] + x.shape[0], axis=0)
                    f["imgs"][-x.shape[0] :] = x
                    if config["labeled_dataset"]:
                        f["labels"].resize(f["labels"].shape[0] + y.shape[0], axis=0)
                        f["labels"][-y.shape[0] :] = y

            if not config["save_images_only"]:
                with h5.File(h5file_name_feats, "a") as f:
                    f["feats"].resize(f["feats"].shape[0] + x_feat.shape[0], axis=0)
                    f["feats"][-x_feat.shape[0] :] = x_feat
                if config["feature_augmentation"]:
                    with h5.File(h5file_name_feats, "a") as f:
                        f["feats_hflip"].resize(
                            f["feats_hflip"].shape[0] + x_feat_hflip.shape[0], axis=0
                        )
                        f["feats_hflip"][-x_feat_hflip.shape[0] :] = x_feat_hflip
        if torch.max(image_id) > config["max_num_image"] :
            break
    print( "Saved index images for evaluation set (in order of appearance in the hdf5 file)" )
    if len(all_image_ids) > 0 :
        np_all_image_ids = np.concatenate(all_image_ids)
    else :
        np_all_image_ids = np.array(all_image_ids[0])
    if len(all_images_filename) > 0:
        np_all_images_filename = np.concatenate(all_images_filename)
    else :
        np_all_images_filename = np.array(all_images_filename[0])
    np.save(npyfile_name, np_all_images_filename[np_all_image_ids],)


def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == "__main__":
    main()
