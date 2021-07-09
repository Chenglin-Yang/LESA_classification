config = dict(
    
    strides=(1,2,2,1),
    wrn=True,
    
    # dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
    # stage_with_dcn=(False, True, True, True), 

    stage_spatial_res=[56, 28, 14, 14], # 224: [56, 28, 14, 14], 1024: [256, 128, 64, 32], 1280: [320, 160, 80, 40]
    stage_with_first_conv = [True, True, True, False],
    
    lesa=dict(
        type='LESA',
        with_cp_UB_terms_only=False,
        pe_type='classification', # ('classification', 'detection_qr')
        groups = 8,
        df_channel_shrink = [2], # df: dynamic fusion
        df_kernel_size = [1,1],
        df_group = [1,1],
    ),
    stage_with_lesa = (False, False, True, True),
)
