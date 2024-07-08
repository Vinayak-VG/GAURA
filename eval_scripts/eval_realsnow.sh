
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname restore_finetune_haze_ownstrengthembed_final_many2one_dark_underwater_noise --run_val --eval_dataset dehazenerfreal --eval_scenes lion --N_samples 128 --num_source_views 10 --typeofmodel finetune_final --pretrained_allweights --viewtrans_depth 8 --rendtrans_depth 8
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname restore_finetune_haze_ownstrengthembed_final_many2many_dark_underwater_noise --run_val --eval_dataset dehazenerfreal --eval_scenes elephant --N_samples 128 --num_source_views 10 --typeofmodel finetune_final --pretrained_allweights --viewtrans_depth 8 --rendtrans_depth 8
# CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_noise_rain_motion_nostrgth_dyndeg_emb_wgt --run_val --eval_dataset realrain --eval_scenes scene1 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realrain 
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene1 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene2 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene4 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene5 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene9 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene10 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth
CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/ablation_3corruption --run_val --eval_dataset realsnow --eval_scenes scene12 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/ablation_3corruption/model_1000000.pth


# CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname restore_finetune_haze_ownstrengthembed_final_many2one_dark_underwater_noise --run_val --eval_dataset dehazenerfreal --eval_scenes lion --N_samples 128 --num_source_views 10 --typeofmodel finetune_final --pretrained_allweights --viewtrans_depth 8 --rendtrans_depth 8
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname restore_finetune_haze_ownstrengthembed_final_many2many_dark_underwater_noise --run_val --eval_dataset dehazenerfreal --eval_scenes elephant --N_samples 128 --num_source_views 10 --typeofmodel finetune_final --pretrained_allweights --viewtrans_depth 8 --rendtrans_depth 8
# CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_noise_rain_motion_nostrgth_dyndeg_emb_wgt --run_val --eval_dataset realrain --eval_scenes scene1 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realrain 
CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/finetune_snow_base_model --run_val --eval_dataset realsnow --eval_scenes scene13 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/finetune_snow_base_model/model_1000000.pth
CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/finetune_snow_base_model --run_val --eval_dataset realsnow --eval_scenes scene14 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/finetune_snow_base_model/model_1000000.pth
CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname /data/cilab4090/restoration_nerf/out/finetune_snow_base_model --run_val --eval_dataset realsnow --eval_scenes scene15 --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name realsnow --ckpt_path /data/cilab4090/restoration_nerf/out/finetune_snow_base_model/model_1000000.pth