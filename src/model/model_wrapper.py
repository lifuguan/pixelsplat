from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from math import ceil
from .types import Gaussians
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

from pathlib import Path
from .ply_export import export_ply
from .encoder.epipolar.conversions import depth_to_relative_disparity
import copy
import numpy as np
import torchvision.utils as save_image

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int


@dataclass
class TestCfg:
    output_path: Path


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass



def random_crop(data,size=[160,224] ,center=None):
    _,_,_,h, w = data['context']['image'].shape
    # size=torch.from_numpy(size)
    batch=copy.deepcopy(data)
    out_h, out_w = size[0], size[1]

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
    # batch['context']['image'] = batch['context']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]
    batch['target']['image'] = batch['target']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]

    # batch['context']['intrinsics'][:,:,0,0]=batch['context']['intrinsics'][:,:,0,0]*w/out_w
    # batch['context']['intrinsics'][:,:,1,1]=batch['context']['intrinsics'][:,:,1,1]*h/out_h
    # batch['context']['intrinsics'][:,:,0,2]=(batch['context']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    # batch['context']['intrinsics'][:,:,1,2]=(batch['context']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h

    batch['target']['intrinsics'][:,:,0,0]=batch['target']['intrinsics'][:,:,0,0]*w/out_w
    batch['target']['intrinsics'][:,:,1,1]=batch['target']['intrinsics'][:,:,1,1]*h/out_h
    batch['target']['intrinsics'][:,:,0,2]=(batch['target']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    batch['target']['intrinsics'][:,:,1,2]=(batch['target']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h



    return batch,center_h,center_w


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.current_step = 0
        self.psnr = 0
        self.automatic_optimization = False
    def batch_cut(self, batch, i):
        return {
            'extrinsics': batch['extrinsics'][:,i:i+2,:,:],
            'intrinsics': batch['intrinsics'][:,i:i+2,:,:],
            'image': batch['image'][:,i:i+2,:,:,:],
            'near': batch['near'][:,i:i+2],
            'far': batch['far'][:,i:i+2],
            'index': batch['index'][:,i:i+2],
        }
        
    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        # gaussians = self.encoder(batch["context"], self.global_step, False)

        opt = self.optimizers()
        opt.zero_grad()
        out_h=160
        out_w=224
        # row=ceil(h/crop_h)
        # col=ceil(w/crop_w)
        # gt_full=data['target']['image'].squeeze(0).squeeze(0)
        # for i in range(row):
        #     for j in range(col):
        #         if i==row-1 and j==col-1:
        #             data_crop=random_crop(  batch,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(w-crop_w//2)))
        #         elif i==row-1:#最后一行
        #             data_crop=random_crop(  batch,size=[crop_h,crop_w],center=(int(h-crop_h//2),int(crop_w//2+j*crop_w)))
        #         elif j==col-1:#z最后一列
        #             data_crop=random_crop( batch,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(w-crop_w//2)))
        #         else:
        #             data_crop=random_crop( batch,size=[crop_h,crop_w],center=(int(crop_h//2+i*crop_h),int(crop_w//2+j*crop_w)))  
        #         # Run the model.
        #         for k in range(batch["context"]["image"].shape[1] - 1):
        #             tmp_batch = self.batch_cut(data_crop["context"],k)
        #             tmp_gaussians = self.encoder(tmp_batch, self.global_step, False)
        #             if k == 0 and i+j==0:
        #                 gaussians: Gaussians = tmp_gaussians
        #             else:
        #                 gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
        #                 gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
        #                 gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
        #                 gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
        with torch.no_grad():
            for i in range(batch["context"]["image"].shape[1] - 1):
                tmp_batch = self.batch_cut(batch["context"],i)
                tmp_gaussians = self.encoder(tmp_batch, self.global_step,features=None)
                # feature_list.append(feature)
                if i == 0:
                    gaussians: Gaussians = tmp_gaussians
                else:
                    gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                    gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                    gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                    gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
            
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
        target_gt = batch["target"]["image"]
        
        output.color.requires_grad_(True)
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)
        self.manual_backward(total_loss)
        rgb_pred_grad=output.color.grad

        row=ceil(h/out_h)
        col=ceil(w/out_w)
        for i in range(row):
            for j in range(col):
                if i==row-1 and j==col-1:
                    data_crop,center_h,center_w=random_crop(  batch,size=[out_h,out_w],center=(int(h-out_h//2),int(w-out_w//2)))
                elif i==row-1:#最后一行
                    data_crop,center_h,center_w=random_crop(  batch,size=[out_h,out_w],center=(int(h-out_h//2),int(out_w//2+j*out_w)))
                elif j==col-1:#z最后一列
                    data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(w-out_w//2)))
                else:
                    data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(out_w//2+j*out_w)))  
                # Run the model.
                data_crop: BatchedExample = self.data_shim(data_crop)
                _, _, _, h_crop, w_crop = data_crop["target"]["image"].shape
                features = self.encoder(batch["context"], self.global_step,features=None,clip_h=4,clip_w=4) 
                for k in range(batch["context"]["image"].shape[1] - 1):
                    tmp_batch = self.batch_cut(data_crop["context"],k)    
                    tmp_gaussians= self.encoder(tmp_batch, self.global_step,features[:,k:k+2,:,:,:],i,j)   #对于其他位置 传入全图feature直接进行计算

                    if k == 0 :
                        gaussians: Gaussians = tmp_gaussians
                    else:
                        gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                        gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                        gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                        gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
                    
                output_1 = self.decoder.forward(
                    gaussians,
                    data_crop["target"]["extrinsics"],
                    data_crop["target"]["intrinsics"],
                    data_crop["target"]["near"],
                    data_crop["target"]["far"],
                    (h_crop, w_crop),
                    depth_mode=self.train_cfg.depth_mode,
                )
                # total_loss = 0
                # for loss_fn in self.losses:
                #     loss = loss_fn.forward(output_1, data_crop, gaussians, self.global_step)
                #     self.log(f"loss/{loss_fn.name}", loss)
                #     total_loss = total_loss + loss
                # self.log("loss/total", total_loss)
                # total_loss.backward(rgb_pred_grad[:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2])
                output_1.color.backward(rgb_pred_grad[:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2])
        # torch.nn.utils.clip_grad_value_(self.parameters(),clip_value=0.5)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")        
        opt.step()
        target_gt = batch["target"]["image"]
        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        # wandb.log("train/psnr_probabilistic", psnr_probabilistic.mean())
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # self.global_step=self.global_step+1
        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}; "
                f"psnr = {psnr_probabilistic.mean():.2f}"
            )
    
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch_, batch_idx):
        batch: BatchedExample = self.data_shim(batch_)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        visualization_dump = {}

        # Render Gaussians.
        # with self.benchmarker.time("encoder"):
        #     gaussians = self.encoder(
        #         batch["context"],
        #         self.global_step,
        #         deterministic=True,
        #         visualization_dump=visualization_dump,
        #     )


        

        with self.benchmarker.time("encoder"):
            for i in range(batch["context"]["image"].shape[1] - 1):
                tmp_batch = self.batch_cut(batch["context"],i)
                tmp_gaussians,_ = self.encoder(tmp_batch, self.global_step)
                if i == 0:
                    gaussians: Gaussians = tmp_gaussians
                else:
                    gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                    gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                    gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                    gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
            
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode="depth",
            )

        # if True:
        #     ply_path = Path(f"outputs/gaussians/fortress/{self.current_step:0>6}.ply")
        #     export_ply(
        #         batch["context"]["extrinsics"][0, 0],
        #         gaussians.means[0],
        #         visualization_dump["scales"][0],
        #         visualization_dump["rotations"][0],
        #         gaussians.harmonics[0],
        #         gaussians.opacities[0],
        #         ply_path,
        #     )

        target_gt = batch["target"]["image"]

        dpt = depth_to_relative_disparity(
                output.depth.squeeze(0).squeeze(0), batch["target"]["near"][0, 0], batch["target"]["far"][0, 0]
            )

        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        if self.global_rank == 0:
            print(
                f"test step {self.current_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"psnr = {psnr_probabilistic.mean():.2f}"
            )
        self.psnr=self.psnr+psnr_probabilistic.mean()
        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        for index, color in zip(batch["target"]["index"][0], output.color[0]):
            save_image(color, path / scene / f"color/{index:0>2}.png")
        self.current_step += 1
        if self.current_step==6:
            print(f'mean_psnr,{self.psnr/self.current_step}')
    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        # gaussians_probabilistic = self.encoder(
        #     batch["context"],
        #     self.global_step,
        #     deterministic=False,
        # )
        for i in range(batch["context"]["image"].shape[1] - 1):
            tmp_batch = self.batch_cut(batch["context"],i)
            tmp_gaussians = self.encoder(tmp_batch, self.global_step, False)
            if i == 0:
                gaussians_probabilistic: Gaussians = tmp_gaussians
            else:
                gaussians_probabilistic.covariances = torch.cat([gaussians_probabilistic.covariances, tmp_gaussians.covariances], dim=1)
                gaussians_probabilistic.means = torch.cat([gaussians_probabilistic.means, tmp_gaussians.means], dim=1)
                gaussians_probabilistic.harmonics = torch.cat([gaussians_probabilistic.harmonics, tmp_gaussians.harmonics], dim=1)
                gaussians_probabilistic.opacities = torch.cat([gaussians_probabilistic.opacities, tmp_gaussians.opacities], dim=1)
        output_probabilistic = self.decoder.forward(
            gaussians_probabilistic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_probabilistic = output_probabilistic.color[0]
        # gaussians_deterministic = self.encoder(
        #     batch["context"],
        #     self.global_step,
        #     deterministic=True,
        # )
        for i in range(batch["context"]["image"].shape[1] - 1):
            tmp_batch = self.batch_cut(batch["context"],i)
            tmp_gaussians = self.encoder(tmp_batch, self.global_step, False)
            if i == 0:
                gaussians_deterministic: Gaussians = tmp_gaussians
            else:
                gaussians_deterministic.covariances = torch.cat([gaussians_deterministic.covariances, tmp_gaussians.covariances], dim=1)
                gaussians_deterministic.means = torch.cat([gaussians_deterministic.means, tmp_gaussians.means], dim=1)
                gaussians_deterministic.harmonics = torch.cat([gaussians_deterministic.harmonics, tmp_gaussians.harmonics], dim=1)
                gaussians_deterministic.opacities = torch.cat([gaussians_deterministic.opacities, tmp_gaussians.opacities], dim=1)
        output_deterministic = self.decoder.forward(
            gaussians_deterministic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_deterministic = output_deterministic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("deterministic", "probabilistic"), (rgb_deterministic, rgb_probabilistic)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            add_label(vcat(*rgb_deterministic), "Target (Deterministic)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = vcat(
            hcat(
                *render_projections(
                    gaussians_probabilistic,
                    256,
                    extra_label="(Probabilistic)",
                )[0]
            ),
            hcat(
                *render_projections(
                    gaussians_deterministic, 256, extra_label="(Deterministic)"
                )[0]
            ),
            align="left",
        )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video, fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
