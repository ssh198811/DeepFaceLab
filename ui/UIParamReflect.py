class UIParam2Config:
    #extract video pic
    b_export_png = True # default use png format

    # model select
    bUseLastModel = True
    bUseNewModel = False
    bUseOldModel = False
    modelname = ""
    # src select
    # todonext
    #dest select
    dest_dir = ""
    # train model select
    bUseSAEHD = True
    # train device select
    bUseGPU = True

    bSaveOnce = False
    bStopTrain = False
    train_state = 0 # 0 not training 1 in training

    #train params SAE
    auto_backup = 0
    write_preview_his = False
    target_itor = 0
    flip_faces = True
    batch_size = 8
    Resolution = 128
    face_type = "f"
    ae_arch = "df"
    auto_encoder_dim = 256
    encoder_dim = 64
    decoder_dim = 64
    decoder_mask_dim = 22
    learn_mask = True
    eyes_priority = False
    place_models_and_optimizer_on_GPU = True
    use_learning_rate_dropout = False
    enable_random_warp_of_samples = True
    GAN_power = 0.0
    true_face_power = 0.0
    face_style_power = 0.0
    background_style_power = 0.0
    color_transfer_for_src_faceset = None
    enable_gradient_clipping = False
    enable_pretrain_model = False

    #merge pic
    merge_model_name = ""
    b_use_interactive_merger = False
    merger_mode = 3
    mask_mode = 1
    erode_mask_modifier = 0
    blur_mask_modifier = 0
    motion_blur_power = 0
    output_face_scale = 0
    color_transfer_mode = None
    super_resolution_power = 0
    image_denoise_power = 0
    bicubic_degrade_power = 0
    color_degrade_power = 0
    sharpen_mode = 0
    sharpen_amount = 0

    #merge video
    bit_rate = 16
    default_fps = 25

class GlobalConfig:
    ffmpeg_cmd_path = ""
    ffprobe_cmd_path = ""

    b_sync_block_op_in_progress = False # 正在进行一个同步阻塞的操作
    b_training_call_in_progress = False