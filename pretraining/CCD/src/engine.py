import torch
import random
from pathlib import Path
from copy import deepcopy
import os
import json
import math
from torch.nn import functional as F
import sys
from Dino.modules import utils


def train_one_epoch(
    config,
    
    train_dataloader,
    train_epoch,
    
    student,
    teacher,
    teacher_without_ddp,
    
    optimizer,
    
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    
    dino_loss,
    
    fp16_scaler,
    
    iteration,
    
    global_epoch
):
    train_dataloader.sampler.set_epoch(train_epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(train_epoch, config.training_epochs)
    
    for (image_tensors, masks, metrics) in metric_logger.log_every(train_dataloader, 10, header):
        epoch = int((iteration + 1) * (config.batch_size_per_gpu * utils.get_world_size()) / config.imgnet_based)
        
        ### examine epoch updating state
        if epoch != global_epoch:
            global_epoch = deepcopy(epoch)
            
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            
            print("Averaged stats:", metric_logger)
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
                # 'args': config,
                'dino_loss': dino_loss.state_dict(),
            }
            
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                
            utils.save_on_master(save_dict, os.path.join(config.output_dir, config.global_name, 'checkpoint.pth'))
            
            if config.saveckp_freq and epoch % config.saveckp_freq == 0:
                utils.save_on_master(save_dict, os.path.join(config.output_dir, config.global_name,
                                                                f'checkpoint{epoch:04}.pth'))
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch}
            
            if utils.is_main_process():
                with (Path(config.output_dir) / f"{config.global_name}/log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Epoch: [{}/{}]'.format(train_epoch, config.training_epochs)

        image_tensors = image_tensors.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[iteration]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[iteration]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            metrics = metrics.float()
            
            student_output = student(image_tensors, metrics, masks, epoch, clusters=None)
            teacher_output = teacher(x=image_tensors, metrics=metrics, target_mask=None, epoch=None, clusters=student_output['zero'], index=student_output['index'])  # only the 2 global views pass through the teacher
            
            affine_grid = F.affine_grid(metrics[:, :2, :], size=(masks.shape[0], 1, masks.shape[1], masks.shape[2]))
            masks_image = F.grid_sample(masks.unsqueeze(1), affine_grid.to(masks.device))
            masks_image = (masks_image > 0.1).float().squeeze()
            
            student_output['gt'] = [masks, masks_image]
            
            loss = dino_loss(student_output, teacher_output, epoch)
            
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        
        if fp16_scaler is None:
            loss.backward()
            if config.clip_grad:
                param_norms = utils.clip_gradients(student, config.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                                config.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if config.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, config.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                                config.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[iteration]  # momentum parameter
            # for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
            #     param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(student.module.backbone.parameters(),
                                        teacher_without_ddp.backbone.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                
            for param_q, param_k in zip(student.module.head.parameters(), teacher_without_ddp.head.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if iteration % config.training_show_iters == 0:
            i = random.randint(0, config.batch_size_per_gpu - 1)
            last_losses = dino_loss.last_losses
            
            for name, loss in last_losses.items():
                scalar_value = loss.data.cpu().numpy()
                tag = 'metric/' + name
                config.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)
                
            lr = optimizer.param_groups[0]["lr"]
            config.writer.add_scalar(tag='metric/' + 'lr', scalar_value=lr, global_step=iteration)
            wd = optimizer.param_groups[0]["weight_decay"]
            config.writer.add_scalar(tag='metric/' + 'wd', scalar_value=wd, global_step=iteration)

        if iteration > config.training_epochs * len(train_dataloader):
            break

        iteration += 1

    return iteration