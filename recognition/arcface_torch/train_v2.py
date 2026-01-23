import argparse
import logging
import os
from datetime import datetime
import socket

import numpy as np
import torch
import torch.nn as nn
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

# PyTorch標準スケジューラ
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, MultiStepLR, SequentialLR, LinearLR
)

# ユーザー定義モジュールのインポート
from backbones import get_model
from dataset import get_dataloader
# ---------------------------------------------------------------------
# [修正1] losses.py から exBoundaryMargin もインポート
# ---------------------------------------------------------------------
from losses import CombinedMarginLoss, BoundaryMargin, exBoundaryMargin

from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging

# ClearMLの確認
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except Exception:
    CLEARML_AVAILABLE = False


assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    # Setup ClearML logging
    if CLEARML_AVAILABLE and rank == 0:
        try:
            hostname = cfg.host_name if cfg.host_name else socket.gethostname()
            head_name = cfg.head_name if cfg.head_name else "ArcFace"
            backbone = cfg.backbone_name if cfg.backbone_name else "ResNet"
            other_tags = cfg.other_tags.replace(' ', '_')
            task_name = f"{hostname}-InsightFace-{head_name}-{backbone}-{other_tags}"
            project_name = cfg.project_name if cfg.project_name else "debug"
            task = Task.init(project_name=project_name, task_name=task_name)
            task.connect(vars(cfg))
            cfg.clearml_task = task
            cfg.clearml_logger = task.get_logger()
            logging.info('ClearML Task initialized: %s' % task_name)
        except Exception as e:
            logging.warning('Failed to initialize ClearML Task: %s' % str(e))
            cfg.clearml_task = None
            cfg.clearml_logger = None
    else:
        cfg.clearml_task = None
        cfg.clearml_logger = None
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
            
    train_loader = get_dataloader(
        cfg,
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, use_se=cfg.use_se).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    # --------------------------------------------------------------------------------
    # Head Selection Logic (PartialFC / BoundaryFace / exBoundaryFace)
    # --------------------------------------------------------------------------------
    head_type = getattr(cfg, "head", "partial_fc")
    
    # 設定ファイルから取得した head_name を使用する
    final_head_type = getattr(cfg, "head_name", head_type) 
    
    # ログ出力 (rank 0 のみで実行)
    if rank == 0:
        logging.info("====================================================")
        logging.info(f"✅ Selected Head Type: **{final_head_type}**")
        
        if final_head_type == "BoundaryFace":
            logging.info(f"   - Boundary s: {getattr(cfg, 'boundary_s', 32.0)}")
            logging.info(f"   - Boundary m: {getattr(cfg, 'boundary_m', 0.5)}")
            logging.info(f"   - Start Epoch: {getattr(cfg, 'boundary_epoch_start', 7)}")
            
        elif final_head_type == "exBoundaryFace":
            logging.info(f"   - Boundary s: {getattr(cfg, 'boundary_s', 32.0)}")
            logging.info(f"   - Boundary m: {getattr(cfg, 'boundary_m', 0.5)}")
            logging.info(f"   - Start Epoch: {getattr(cfg, 'boundary_epoch_start', 30)}")
            logging.info(f"   - K_nn: {getattr(cfg, 'boundary_K_nn', 3)}")
            logging.info(f"   - Tau OSN: {getattr(cfg, 'boundary_tau_osn', 0.25)}")
            
        elif final_head_type == "partial_fc":
            logging.info(f"   - Margin List (m1, m2, m3): {cfg.margin_list}")
        logging.info("====================================================")
    
    # head_type の変数名は元のコードに合わせて保持しておく
    head_type = final_head_type


    module_partial_fc = None
    boundary_header = None
    criterion_ce = None  # BoundaryFace用
    params_to_optimize = []

    if head_type == "partial_fc":
        logging.info("Init Head: Partial FC V2")
        margin_loss = CombinedMarginLoss(
            64,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold
        )
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        
        params_to_optimize = [
            {"params": backbone.parameters()}, 
            {"params": module_partial_fc.parameters()}
        ]

    elif head_type == "BoundaryFace":
        logging.info("Init Head: BoundaryFace (Standard)")
        boundary_s = getattr(cfg, "boundary_s", 32.0)
        boundary_m = getattr(cfg, "boundary_m", 0.5)
        boundary_epoch_start = getattr(cfg, "boundary_epoch_start", 7)

        boundary_header = BoundaryMargin(
            in_feature=cfg.embedding_size,
            out_feature=cfg.num_classes,
            s=boundary_s,
            m=boundary_m,
            easy_margin=False,
            epoch_start=boundary_epoch_start
        ).cuda()

        boundary_header = torch.nn.parallel.DistributedDataParallel(
            module=boundary_header, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16
        )
        boundary_header.train()
        
        criterion_ce = nn.CrossEntropyLoss().cuda()

        params_to_optimize = [
            {"params": backbone.parameters()}, 
            {"params": boundary_header.parameters()}
        ]
        
    # ---------------------------------------------------------------------
    # [修正2] exBoundaryFace の初期化分岐を追加
    # ---------------------------------------------------------------------
    elif head_type == "exBoundaryFace":
        logging.info("Init Head: exBoundaryFace (Extended with OSN/CSN filtering)")
        # Configから読み込み（デフォルト値付き）
        boundary_s = getattr(cfg, "boundary_s", 32.0)
        boundary_m = getattr(cfg, "boundary_m", 0.5)
        boundary_epoch_start = getattr(cfg, "boundary_epoch_start", 30) # ノイズ除去開始は少し遅め推奨
        boundary_K_nn = getattr(cfg, "boundary_K_nn", 3)
        boundary_tau_osn = getattr(cfg, "boundary_tau_osn", 0.25)

        boundary_header = exBoundaryMargin(
            in_feature=cfg.embedding_size,
            out_feature=cfg.num_classes,
            s=boundary_s,
            m=boundary_m,
            easy_margin=False,
            epoch_start=boundary_epoch_start,
            K_nn=boundary_K_nn,
            tau_osn=boundary_tau_osn
        ).cuda()

        # DDPでラップ
        boundary_header = torch.nn.parallel.DistributedDataParallel(
            module=boundary_header, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16
        )
        boundary_header.train()
        
        criterion_ce = nn.CrossEntropyLoss().cuda()

        params_to_optimize = [
            {"params": backbone.parameters()}, 
            {"params": boundary_header.parameters()}
        ]
        
    else:
        raise ValueError(f"Unknown head type: {head_type}")


    # --------------------------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------------------------
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=params_to_optimize,
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=params_to_optimize,
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    # --------------------------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------------------------
    steps_per_epoch = cfg.num_image // cfg.total_batch_size
    scheduler_name = getattr(cfg, "scheduler", "Polynomial")

    if scheduler_name == "Polynomial":
        logging.info(f"Using PolynomialLRWarmup scheduler (Warmup integrated).")
        poly_power = getattr(cfg, "poly_power", 1.0)
        lr_scheduler = PolynomialLRWarmup(
            optimizer=opt,
            warmup_iters=cfg.warmup_step,
            total_iters=cfg.total_step,
            power=poly_power
        )

    elif scheduler_name in ["Cosine", "Multistep"]:
        if cfg.warmup_step == 0:
            logging.info(f"Using {scheduler_name} scheduler (No Warmup).")
            if scheduler_name == "Cosine":
                lr_scheduler = CosineAnnealingLR(optimizer=opt, T_max=cfg.total_step, eta_min=0.0)
            else: 
                epoch_milestones = getattr(cfg, "epoch_milestones", [cfg.num_epoch // 2, (3 * cfg.num_epoch) // 4])
                step_milestones = [e * steps_per_epoch for e in epoch_milestones]
                gamma = getattr(cfg, "gamma", 0.1)
                lr_scheduler = MultiStepLR(optimizer=opt, milestones=step_milestones, gamma=gamma)
        else:
            logging.info(f"Using {scheduler_name} scheduler with Linear Warmup.")
            warmup_scheduler = LinearLR(optimizer=opt, start_factor=0.01, total_iters=cfg.warmup_step)
            if scheduler_name == "Cosine":
                main_scheduler = CosineAnnealingLR(optimizer=opt, T_max=cfg.total_step - cfg.warmup_step, eta_min=0.0)
            else: 
                epoch_milestones = getattr(cfg, "epoch_milestones", [cfg.num_epoch // 2, (3 * cfg.num_epoch) // 4])
                step_milestones = [e * steps_per_epoch for e in epoch_milestones]
                adjusted_milestones = [m - cfg.warmup_step for m in step_milestones if m > cfg.warmup_step]
                gamma = getattr(cfg, "gamma", 0.1)
                main_scheduler = MultiStepLR(optimizer=opt, milestones=adjusted_milestones, gamma=gamma)

            lr_scheduler = SequentialLR(
                optimizer=opt,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.warmup_step] 
            )
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' is not recognized.")


    # --------------------------------------------------------------------------------
    # Resume & Main Loop
    # --------------------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        
        if head_type == "partial_fc":
            module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        elif head_type == "BoundaryFace":
            boundary_header.module.load_state_dict(dict_checkpoint["state_dict_boundary"])
        # [修正3] exBoundaryFace の Resume 対応
        elif head_type == "exBoundaryFace":
            # 互換性のため state_dict_boundary キーを使用するか、独自キーを使用
            if "state_dict_ex_boundary" in dict_checkpoint:
                boundary_header.module.load_state_dict(dict_checkpoint["state_dict_ex_boundary"])
            elif "state_dict_boundary" in dict_checkpoint:
                boundary_header.module.load_state_dict(dict_checkpoint["state_dict_boundary"])
            
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger,
        clearml_logger = cfg.clearml_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        
        # DataLoader Loop
        for _, batch in enumerate(train_loader):
            global_step += 1
            
            # ---------------------------------------------------------------------
            # [重要] 画像パスの取得と None チェックへの対応
            # dataset.py がパスを返さない(len=2)場合は、BoundaryFaceのログ出力はスキップされます。
            # ログを出したい場合は dataset.py を修正してパスを返すようにしてください。
            # ---------------------------------------------------------------------
            if len(batch) == 3:
                img, local_labels, img_paths = batch
            else:
                img, local_labels = batch
                img_paths = None 

            img = img.cuda(non_blocking=True)
            local_labels = local_labels.cuda(non_blocking=True)
            
            local_embeddings = backbone(img)
            loss = 0

            # --- Loss Calculation Switch ---
            if head_type == "partial_fc":
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            
            elif head_type == "BoundaryFace":
                # BoundaryMargin (Standard)
                output, final_loss_term, rectified_label = boundary_header(
                    local_embeddings, local_labels, epoch, img_paths, cfg.output
                )
                loss = criterion_ce(output, rectified_label) + final_loss_term

            # ---------------------------------------------------------------------
            # [修正4] exBoundaryFace の Forward 処理 (img_paths を渡す)
            # ---------------------------------------------------------------------
            elif head_type == "exBoundaryFace":
                # exBoundaryMargin (Extended)
                # 引数は BoundaryFace と同じインターフェースを持つように設計されている
                output, final_loss_term, rectified_label = boundary_header(
                    local_embeddings, local_labels, epoch, img_paths, cfg.output
                )
                loss = criterion_ce(output, rectified_label) + final_loss_term


            # Backward
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            # Logging
            with torch.no_grad():
                if wandb_logger:
                    log_dict = {
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    }
                    if head_type in ["BoundaryFace", "exBoundaryFace"]:
                         log_dict['Loss/Boundary_Term'] = final_loss_term.item()
                    wandb_logger.log(log_dict)
                
                # ClearML Logging
                if rank == 0 and cfg.clearml_logger and (global_step % 500 == 0):
                    try:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        cfg.clearml_logger.report_scalar(
                            'Train', 'loss', iteration=global_step, value=loss.item()
                        )
                        cfg.clearml_logger.report_scalar(
                            'LR', 'learning_rate', iteration=global_step, value=current_lr
                        )
                        if head_type in ["BoundaryFace", "exBoundaryFace"]:
                             cfg.clearml_logger.report_scalar(
                                 'Train', 'Boundary_Term', iteration=global_step, value=final_loss_term.item()
                             )
                    except Exception as e:
                        logging.warning('ClearML report_scalar failed: %s' % str(e))

                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
                
        if rank == 0: # 検証とログは rank 0 でのみ実行
            logging.info(f"--- Running End-of-Epoch Verification (Epoch: {epoch+1}) ---")
            callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            if head_type == "partial_fc":
                checkpoint["state_dict_softmax_fc"] = module_partial_fc.state_dict()
            elif head_type == "BoundaryFace":
                checkpoint["state_dict_boundary"] = boundary_header.module.state_dict()
            elif head_type == "exBoundaryFace":
                # [修正5] exBoundaryFace 用の保存キー
                checkpoint["state_dict_ex_boundary"] = boundary_header.module.state_dict()
                
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:            
            epoch_model_name = f"model_epoch_{epoch + 1}.pt"
            path_module = os.path.join(cfg.output, epoch_model_name)
            
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch + 1}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

            if cfg.clearml_task:
                    try:
                        artifact_name = f"model_epoch_{epoch + 1}"
                        cfg.clearml_task.upload_artifact(
                            artifact_name, artifact_object=path_module
                        )
                    except Exception as e:
                        logging.warning('ClearML upload_artifact failed: %s' % str(e))
         
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

        if cfg.clearml_task:
            try:
                cfg.clearml_task.upload_artifact(
                    "model_final", artifact_object=path_module
                )
            except Exception as e:
                logging.warning('ClearML upload_artifact (final) failed: %s' % str(e))
      

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())