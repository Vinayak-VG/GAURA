# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_snow_base_model_lessdata --run_val --eval_dataset llff_render --eval_scenes scene9 --llffhold 8 --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_snow_base_model_lessdata/model_990000.pth
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes scene16 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_defocus_base_lessdata --run_val --eval_dataset llff_render --eval_scenes defocuscoral --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_defocus_base_lessdata/model_961000.pth --render_stride 1 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes blurstair --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_defocus_base_lessdata --run_val --eval_dataset llff_render --eval_scenes defocuscaps --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_defocus_base_lessdata/model_961000.pth --render_stride 1 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes W001 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes blurstair --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 64

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset nerf_synthetic_render --eval_scenes buu --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 64

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes blurball --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 128
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes blurgirl --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 128

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes rain_scene11 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 128
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes rain_rb3 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 128

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes haze_J005 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 128
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes haze_L003 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes haze_L004 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 64
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes haze_W002 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 1 --N_samples 64

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str --run_val --eval_dataset llff_render --eval_scenes dark_still3 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/final_genexpt_dark_haze_rain_motion_yesstrgth_dyndeg_emb_wgt_patch_vggloss_str/model_910000.pth --render_stride 2 --N_samples 64

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_defocus_base_lessdata --run_val --eval_dataset llff_render --eval_scenes defocuscaps --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_defocus_base_lessdata/model_961000.pth --render_stride 1 --N_samples 128
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_defocus_base_lessdata --run_val --eval_dataset llff_render --eval_scenes defocusseal --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_defocus_base_lessdata/model_961000.pth --render_stride 1 --N_samples 128

# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_snow_base_model_lessdata --run_val --eval_dataset llff_render --eval_scenes snow_scene13 --llffhold 8 --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_snow_base_model_lessdata/model_990000.pth --render_stride 1 --N_samples 128
# CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finetune_snow_base_model_lessdata --run_val --eval_dataset llff_render --eval_scenes snow_scene14 --llffhold 8 --folder_name render_video --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf/out/finetune_snow_base_model_lessdata/model_990000.pth --render_stride 1 --N_samples 128

CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss --run_val --eval_dataset llff_render --eval_scenes still2 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf_finalsprint/out/finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss/model_1120000.pth --render_stride 2 --N_samples 64
CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss --run_val --eval_dataset llff_render --eval_scenes still3 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf_finalsprint/out/finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss/model_1120000.pth --render_stride 2 --N_samples 64
CUDA_VISIBLE_DEVICES=0 python3 -W ignore render_gsn.py --config configs/transibr_bigger_full.txt --expname finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss --run_val --eval_dataset llff_render --eval_scenes still4 --llffhold 8 --folder_name render_video --typeofmodel yesstrgth_dyndeg_emb_wgt_strenc --viewtrans_depth 8 --rendtrans_depth 8 --ckpt_path /home/vinayak/restoration_nerf_finalsprint/out/finalpaper_genexpt_5corr_yesstrgth_dyndeg_emb_wgt_patch_vggloss_withclean_emb_resnet_strenc_noregloss/model_1120000.pth --render_stride 2 --N_samples 64