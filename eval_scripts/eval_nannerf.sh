CUDA_VISIBLE_DEVICES=0 python3 -W ignore eval_transibr.py --config configs/transibr_bigger_full.txt --expname final_genexpt_dark_noise_rain_motion_nostrgth_dyndeg_emb_wgt --run_val --eval_dataset nannerf --eval_scenes stairs --N_samples 64 --typeofmodel nostrgth_dyndeg_emb_wgt --viewtrans_depth 8 --rendtrans_depth 8 --folder_name nannerf --render_stride 2
