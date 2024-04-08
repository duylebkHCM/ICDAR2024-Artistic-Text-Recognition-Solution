# more info at outputs/parseq_custom/2024-03-23_16-07-33/config/overrides.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py --checkpoint outputs/parseq_custom/2024-03-23_16-07-33/checkpoints/last.ckpt \
--root_dir /dataset/

# more info at outputs/parseq_custom/2024-03-27_12-33-11/config/overrides.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py --checkpoint outputs/parseq_custom/2024-03-27_12-33-11/checkpoints/swa_epoch678.ckpt \
--root_dir /dataset/

# more info at outputs/parseq_viptr/2024-03-31_02-24-43/config/overrides.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py --checkpoint outputs/parseq_viptr/2024-03-31_02-24-43/checkpoints/last.ckpt \
--root_dir /dataset/

# more info at outputs/parseq_viptr/2024-04-01_03-21-42/config/overrides.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py --checkpoint outputs/parseq_viptr/2024-04-01_03-21-42/checkpoints/last.ckpt \
--root_dir /dataset/
