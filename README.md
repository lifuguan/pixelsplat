
'''
注意wandb.mode=online，调用自己的名字账号在main.yaml里
'''


export CUDA_VISIBLE_DEVICES=0
python -m src.main wandb.mode=online \
      +experiment=waymo \
      data_loader.train.num_workers=0 \
      data_loader.test.num_workers=0 \
      data_loader.val.num_workers=0 \
      data_loader.train.batch_size=1 \
      ++dataset.scene=['019'] // 是否逐场景
      // checkpointing.load= // 预训练权重



export CUDA_VISIBLE_DEVICES=1
python -m src.main wandb.mode=online \
      +experiment=waymo \
      data_loader.train.num_workers=0 \
      data_loader.test.num_workers=0 \
      data_loader.val.num_workers=0 \
      data_loader.train.batch_size=1 \
      // checkpointing.load= // 预训练权重