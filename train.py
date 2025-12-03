# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys

import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
import copy
from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.transformer_network import Embedder
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
import imageio
import cv2
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    pos_enc = None
    view_enc = None
    model = IBRNetModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    
    # print(model.feature_net.promptblock1.prompt1.prompt_param.weight.data[5][0])
    # model.feature_net.promptblock1.prompt1.prompt_param.weight.data[5] = copy.deepcopy((model.feature_net.promptblock1.prompt1.prompt_param.weight.data[1] + model.feature_net.promptblock1.prompt1.prompt_param.weight.data[2] + model.feature_net.promptblock1.prompt1.prompt_param.weight.data[3] + model.feature_net.promptblock1.prompt1.prompt_param.weight.data[4])/4) 
    # model.feature_net.promptblock2.prompt1.prompt_param.weight.data[5] = copy.deepcopy((model.feature_net.promptblock2.prompt1.prompt_param.weight.data[1] + model.feature_net.promptblock2.prompt1.prompt_param.weight.data[2] + model.feature_net.promptblock2.prompt1.prompt_param.weight.data[3] + model.feature_net.promptblock2.prompt1.prompt_param.weight.data[4])/4) 
    # model.feature_net.promptblock3.prompt1.prompt_param.weight.data[5] = copy.deepcopy((model.feature_net.promptblock3.prompt1.prompt_param.weight.data[1] + model.feature_net.promptblock3.prompt1.prompt_param.weight.data[2] + model.feature_net.promptblock3.prompt1.prompt_param.weight.data[3] + model.feature_net.promptblock3.prompt1.prompt_param.weight.data[4])/4) 
    # print(model.feature_net.promptblock1.prompt1.prompt_param.weight.data[5][0])
    
    for name, param in model.net_coarse.named_parameters():
        if param.requires_grad == True:
            print(name)
    for name, param in model.feature_net.named_parameters():
        if param.requires_grad == True:
            print(name)

    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0

    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            if args.sample_mode == "center":
                N_rand = args.N_rand
            else:
                N_rand = int(
                    1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
                )
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )
            
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2), embed_id=train_data["embed_id1"].to(device))
            # ray_batch["clean_rgbs"] = ray_batch["clean_rgbs"][0].permute(0, 3, 1, 2)
            # ray_batch["clean_rgbs"] = torch.nn.functional.interpolate(ray_batch["clean_rgbs"], size = (featmaps[1].shape[2], featmaps[1].shape[3]), mode='bilinear')
            
            ret = render_rays(
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                N_importance=args.N_importance,
                det=args.det,
                white_bkgd=args.white_bkgd,
                net=args.net,
                pos_enc=pos_enc,
                view_enc=view_enc,
                transret_alpha=args.N_importance > 0,
                transsep_fine=args.transsep_fine,
                embed_id1=train_data["embed_id1"].to(device),
            )

            # if global_step % 25 == 0:
            #     print(model.feature_net.promptblock2.prompt1.prompt_param.weight[5][0].item(), model.feature_net.promptblock2.prompt1.prompt_param.weight[5][5].item(), model.feature_net.promptblock2.prompt1.prompt_param.weight[5][10].item(), model.feature_net.promptblock2.prompt1.prompt_param.weight[5][15].item())

            model.optimizer.zero_grad()
            
            loss, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, featmaps[-1], train_data["target"].to(device), scalars_to_log)
                
            if ret["outputs_fine"] is not None:
                fine_loss, scalars_to_log = criterion(
                    ret["outputs_fine"], ray_batch, scalars_to_log
                )
                loss += fine_loss

            loss.backward()
            scalars_to_log["loss"] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                    scalars_to_log["train/coarse-loss"] = mse_error
                    scalars_to_log["train/coarse-psnr-training-batch"] = mse2psnr(mse_error)
                    if ret["outputs_fine"] is not None:
                        mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/fine-loss"] = mse_error
                        scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)

                    logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)
                    print("each iter time {:.05f} seconds".format(dt))

                if global_step % args.i_weights == 0:
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    print("Logging a random validation view...")
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(
                        val_data, device, render_stride=args.render_stride
                    )
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_sampler,
                        projector,
                        gt_img,
                        render_stride=args.render_stride,
                        prefix="val/",
                        out_folder=out_folder,
                        net=args.net,
                        pos_enc=pos_enc,
                        view_enc=view_enc,
                        transret_alpha=args.N_importance > 0,
                        transsep_fine=args.transsep_fine,
                        embed_id1=val_data["embed_id1"].to(device),
                        # embed_id2=val_data["embed_id2"].to(device)
                    )
                    torch.cuda.empty_cache()

                    print("Logging current training view...")
                    tmp_ray_train_sampler = RaySamplerSingleImage(
                        train_data, device, render_stride=1
                    )
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_train_sampler,
                        projector,
                        gt_img,
                        render_stride=1,
                        prefix="train/",
                        out_folder=out_folder,
                        net=args.net,
                        pos_enc=pos_enc,
                        view_enc=view_enc,
                        transret_alpha=args.N_importance > 0,
                        transsep_fine=args.transsep_fine,
                        embed_id1=train_data["embed_id1"].to(device),
                        # embed_id2=train_data["embed_id2"].to(device)
                    )
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    net="mlp_ibr",
    pos_enc=None,
    view_enc=None,
    transret_alpha=False,
    transsep_fine=False,
    embed_id1=None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2), embed_id=embed_id1)
        else:
            featmaps = [None, None]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            net=net,
            pos_enc=pos_enc,
            view_enc=view_enc,
            transret_alpha=transret_alpha,
            transsep_fine=transsep_fine,
            embed_id1=embed_id1,
        )

    # average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    average_im = ray_sampler.src_rgbs.cpu()[0][0]
    # from PIL import Image
    # for i in range(ray_sampler.src_rgbs.cpu().shape[1]):
    #     image = ray_sampler.src_rgbs.cpu()[0][i].numpy()*255.
    #     image = image.astype('uint8')
    #     image = Image.fromarray(image)
    #     image.save(f"test/{i}_src_views.png")
    # # gt_img = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu()).permute(1, 2, 0).numpy()*255.
    # gt_img = np.clip((gt_img.numpy()*255).astype('uint8'), 0, 255)
    # gt_img = Image.fromarray(gt_img)
    # gt_img.save(f"test/{i}_target_view.png")
    # quit()

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    else:
        depth_im = None

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, ret["outputs_fine"]["depth"].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, (np.clip(rgb_im, 0, 1)*255).astype(np.uint8))
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None
        else ret["outputs_coarse"]["rgb"]
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    model.switch_to_train()

if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        # torch.set_num_threads(1)
        cv2.setNumThreads(1)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)