from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Layer

def sensor_encoder(x):
    """Two Conv1D layers with Max Pooling removed to preserve the 125 timesteps."""
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x   

def position_encoder(x):
    """Single Conv1D."""
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x   
    
def device_encoder(x):
    """Un Conv1D pequeño porque solo son 3 variables booleanas."""
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x
    
def gamerot_encoder(x):
    """Dedicated encoder for GameRotation angular distance sequence."""
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def accel_encoder(x):
    """Small encoder for acceleration stream — mean, peak, brake flag."""
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def film_layer(sensor_features, conditioning_signal):
    units = sensor_features.shape[-1]
    gamma = layers.Dense(units, activation='linear')(conditioning_signal)  # scale
    beta  = layers.Dense(units, activation='linear')(conditioning_signal)  # shift
    return layers.Multiply()([sensor_features, gamma]) + beta
    
def gated_film_layer(x, conditioning_signal):
    gamma = layers.Dense(x.shape[-1])(conditioning_signal)
    beta  = layers.Dense(x.shape[-1])(conditioning_signal)
    gate  = layers.Dense(1, activation='sigmoid')(conditioning_signal)
    
    gamma = 1.0 + gate * gamma
    beta  = gate * beta

    return layers.Multiply()([x, gamma]) + beta

def gated_film_layer_dropout(x, context, dropout_rate=0.6):
    # ── stream-level dropout: Keras handles training/inference automatically ─
    context = layers.Dropout(rate=dropout_rate)(context)

    # ── gated FiLM ───────────────────────────────────────────────────────────
    gate  = layers.Dense(1,            activation='sigmoid')(context)  # (T, 1)
    gamma = layers.Dense(x.shape[-1])(context)                         # (T, 96)
    beta  = layers.Dense(x.shape[-1])(context)                         # (T, 96)

    gamma = 1.0 + gate * gamma
    beta  = gate * beta

    return layers.Multiply()([x, gamma]) + beta

def weak_film_layer(x, context, gamma_scale=0.1):
    gamma = layers.Dense(x.shape[-1])(context)
    beta  = layers.Dense(x.shape[-1])(context)
    
    gamma = 1.0 + gamma_scale * gamma   # small perturbation around identity
    beta  = gamma_scale * beta

    return layers.Multiply()([x, gamma]) + beta

def build_model_v4(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),    name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),    name="device_data") 
    
    enc_sens = sensor_encoder(inp_sensor)      
    enc_pos  = position_encoder(inp_position)  
    enc_dev  = device_encoder(inp_device)

    # ── merge ─────────────────────────────────────────────────
    x = layers.concatenate([enc_sens, enc_pos, enc_dev])

    # ── temporal refinement ───────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention ───────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ───────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── classification head ───────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    model = Model(
        inputs  = [inp_sensor, inp_position, inp_device],
        outputs = out,
    )
    return model

def build_model_v5(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),    name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),    name="device_data")

    enc_sens = sensor_encoder(inp_sensor)
    enc_dev  = device_encoder(inp_device)

    speed_ctx = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_position)

    # A velocidade não entra juntamente com os sensores 
    # "The speed value never enters the classification pathway directly — it can only influence the model by modulating the sensor branch.
    #  You're constraining how the model is allowed to use speed."
    # FiLM: speed modulates sensor encoding
    enc_sens_conditioned = film_layer(enc_sens, speed_ctx)

    x = layers.concatenate([enc_sens_conditioned, enc_dev])

    # ── temporal refinement ───────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention ───────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ───────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── classification head ───────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    model = Model(
        inputs  = [inp_sensor, inp_position, inp_device],
        outputs = out,
    )
    return model

def build_model_v6(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),    name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),    name="device_data")

    enc_sens = sensor_encoder(inp_sensor)
    enc_dev  = device_encoder(inp_device)          # kept, used later

    # ── speed context (FiLM conditioner only, never enters classifier) ──
    speed_ctx = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_position)

    # ── FiLM: speed modulates sensor encoding ──────────────────────────
    enc_sens_conditioned = gated_film_layer_dropout(enc_sens, speed_ctx)

    # ── temporal refinement (sensors only, no device yet) ──────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_sens_conditioned)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention ────────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: inject device state after temporal reasoning ───────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    model = Model(
        inputs  = [inp_sensor, inp_position, inp_device],
        outputs = out,
    )
    return model   

def build_model_v7(n_timesteps, n_sensor_feat, n_gamerot_feat, n_pos_feat, n_dev_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
    inp_gamerot  = Input(shape=(n_timesteps, n_gamerot_feat), name="gamerot_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),     name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

    enc_sens    = sensor_encoder(inp_sensor)
    enc_gamerot = gamerot_encoder(inp_gamerot)          # 125 × 32
    enc_dev     = device_encoder(inp_device)

    # merge before FiLM so speed can modulate both streams
    enc_combined = layers.concatenate([enc_sens, enc_gamerot])  # 125 × 96

    # ── speed context (unchanged) ───────────────────────────────────
    speed_ctx = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_position)

    # ── FiLM: speed modulates combined sensor+gamerot encoding ───────
    enc_conditioned = gated_film_layer_dropout(enc_combined, speed_ctx)

    # ── temporal refinement ─────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_conditioned)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention (unchanged) ────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack (unchanged) ────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state (unchanged) ───────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head (unchanged) ─────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  =[inp_sensor, inp_position, inp_gamerot, inp_device],
        outputs=out,
    )

def build_model_v8(n_timesteps, n_sensor_feat, n_gamerot_feat,
                   n_accel_feat, n_pos_feat, n_dev_feat):

    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
    inp_gamerot  = Input(shape=(n_timesteps, n_gamerot_feat), name="gamerot_data")
    inp_speed    = Input(shape=(n_timesteps, n_pos_feat),     name="speed_data")    
    inp_accel    = Input(shape=(n_timesteps, n_accel_feat),   name="accel_data")    
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

    enc_sens    = sensor_encoder(inp_sensor)               # (T/4, 64)
    enc_gamerot = gamerot_encoder(inp_gamerot)             # (T/4, 32)
    enc_accel   = accel_encoder(inp_accel)                 # (T/4, 16)
    enc_dev     = device_encoder(inp_device)               # (T/4, 16)

    # ── merge sensor streams ────────────────────────────────────────
    enc_combined = layers.concatenate([enc_sens, enc_gamerot])  # (T/4, 96)

    # ── FiLM: speed modulates sensor+gamerot only ───────────────────
    speed_ctx        = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_speed)
    enc_conditioned  = film_layer(enc_combined, speed_ctx)      # (T/4, 96)

    # ── temporal refinement + inject accel directly ─────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_conditioned)
    x = layers.BatchNormalization()(x)
    x = layers.concatenate([x, enc_accel])                      # (T/4, 80)

    # ── multi-head attention ────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state ───────────────────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  = [inp_sensor, inp_gamerot, inp_speed, inp_accel, inp_device],
        outputs = out,
    )   

def build_model_v9(n_timesteps, n_sensor_feat, n_gamerot_feat,
                   n_accel_feat, n_dev_feat):

    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
    inp_gamerot  = Input(shape=(n_timesteps, n_gamerot_feat), name="gamerot_data")
    inp_accel    = Input(shape=(n_timesteps, n_accel_feat),   name="accel_data")    
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

    enc_sens    = sensor_encoder(inp_sensor)               # (T/4, 64)
    enc_gamerot = gamerot_encoder(inp_gamerot)             # (T/4, 32)
    enc_accel   = accel_encoder(inp_accel)                 # (T/4, 16)
    enc_dev     = device_encoder(inp_device)               # (T/4, 16)

    # ── merge sensor streams ────────────────────────────────────────
    enc_combined = layers.concatenate([enc_sens, enc_gamerot])  # (T/4, 96)

    # ── temporal refinement + inject accel directly ─────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_combined)
    x = layers.BatchNormalization()(x)
    x = layers.concatenate([x, enc_accel])                      # (T/4, 80)

    # ── multi-head attention ────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state ───────────────────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  = [inp_sensor, inp_gamerot, inp_accel, inp_device],
        outputs = out,
    )   


def build_model_v10(n_timesteps, n_sensor_feat, n_gamerot_feat,
                   n_accel_feat, n_dev_feat):

    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
    inp_gamerot  = Input(shape=(n_timesteps, n_gamerot_feat), name="gamerot_data")
    inp_accel    = Input(shape=(n_timesteps, n_accel_feat),   name="accel_data")    
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

    enc_sens    = sensor_encoder(inp_sensor)               # (T/4, 64)
    enc_gamerot = gamerot_encoder(inp_gamerot)             # (T/4, 32)
    enc_accel   = accel_encoder(inp_accel)                 # (T/4, 16)
    enc_dev     = device_encoder(inp_device)               # (T/4, 16)

    # ── merge sensor streams ────────────────────────────────────────
    enc_combined = layers.concatenate([enc_sens, enc_gamerot])  # (T/4, 96)

    # ── temporal refinement + inject accel directly ─────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_combined)
    x = layers.BatchNormalization()(x)
    x = layers.concatenate([x, enc_accel])                      # (T/4, 80)

    # ── multi-head attention ────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state ───────────────────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  = [inp_sensor, inp_gamerot, inp_accel, inp_device],
        outputs = out,
    )   

def build_model_v12(n_timesteps, n_sensor_feat,
                   n_accel_feat, n_pos_feat, n_dev_feat):

    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
    inp_speed    = Input(shape=(n_timesteps, n_pos_feat),     name="speed_data")    
    inp_accel    = Input(shape=(n_timesteps, n_accel_feat),   name="accel_data")    
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

    enc_sens    = sensor_encoder(inp_sensor)               # (T/4, 64)
    enc_accel   = accel_encoder(inp_accel)                 # (T/4, 16)
    enc_dev     = device_encoder(inp_device)               # (T/4, 16)

    # ── FiLM: speed modulates sensor+gamerot only ───────────────────
    speed_ctx        = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_speed)
    enc_conditioned  = gated_film_layer(enc_sens, speed_ctx)      # (T/4, 96)

    # ── temporal refinement + inject accel directly ─────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_conditioned)
    x = layers.BatchNormalization()(x)
    x = layers.concatenate([x, enc_accel])                      # (T/4, 80)

    # ── multi-head attention ────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state ───────────────────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  = [inp_sensor, inp_speed, inp_accel, inp_device],
        outputs = out,
    )

def build_model_v14(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat):
    """
    v14 — refinement of v7.2 (the best PR-AUC model, 0.9611) with ONE change:
      - GAMEROT_angular_distance is folded into the sensor stream (use suffix .3)
        instead of getting its own encoder — CVPFI showed it ~0 as a standalone
        stream while costing encoder capacity.

    The rest of the architecture is kept identical to v7.2 (gated-FiLM, 4-head
    attention, stacked BiLSTM(64)->BiLSTM(32), Dense(32) head) so the only
    variables under test are the folded sensor stream and the manual labels.

    No acceleration stream: instant_acceleration is the derivative of speed_mps
    (same GPS source/rate) and is recoverable by the speed Conv1D, while the
    engineered accel channels (peak/hard-brake/hard-accel) are the spurious
    urban-context proxies we want to avoid. Speed conditions the sensor encoding
    via gated FiLM and never enters the classification pathway directly.

    Inputs: sensor_data, speed_data, device_data.
    """
    inp_sensor = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_speed  = Input(shape=(n_timesteps, n_pos_feat),    name="speed_data")
    inp_device = Input(shape=(n_timesteps, n_dev_feat),    name="device_data")

    enc_sens = sensor_encoder(inp_sensor)                  # (T, 64); GAMEROT folded in
    enc_dev  = device_encoder(inp_device)                  # (T, 16)

    # ── FiLM: speed modulates the sensor encoding, never votes directly ──
    speed_ctx       = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_speed)
    enc_conditioned = gated_film_layer_dropout(enc_sens, speed_ctx)         # (T, 64)

    # ── temporal refinement ─────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_conditioned)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention ────────────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack (kept from v7.2) ───────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # ── late fusion: device state ───────────────────────────────────
    enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
    x = layers.concatenate([x, enc_dev_flat])

    # ── classification head ─────────────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    return Model(
        inputs  = [inp_sensor, inp_speed, inp_device],
        outputs = out,
    )

def build_model_v13(n_timesteps, n_sensor_feat, n_gamerot_feat,
                   n_accel_feat, n_pos_feat, n_dev_feat):

        inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat),  name="sensor_data")
        inp_gamerot  = Input(shape=(n_timesteps, n_gamerot_feat), name="gamerot_data")
        inp_speed    = Input(shape=(n_timesteps, n_pos_feat),     name="speed_data")    
        inp_accel    = Input(shape=(n_timesteps, n_accel_feat),   name="accel_data")    
        inp_device   = Input(shape=(n_timesteps, n_dev_feat),     name="device_data")

        enc_sens    = sensor_encoder(inp_sensor)               # (T/4, 64)
        enc_gamerot = gamerot_encoder(inp_gamerot)             # (T/4, 32)
        enc_accel   = accel_encoder(inp_accel)                 # (T/4, 16)
        enc_dev     = device_encoder(inp_device)               # (T/4, 16)

        # ── merge sensor streams ─────────────────────────────────────────
        enc_combined = layers.concatenate([enc_sens, enc_gamerot])        # (T/4, 96)

        # ── build a joint context from speed + accel ─────────────────────
        speed_ctx = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_speed)
        accel_ctx = layers.Conv1D(16, kernel_size=5, padding='same', activation='relu')(inp_accel)
        joint_ctx = layers.concatenate([speed_ctx, accel_ctx])            # (T, 32)
        joint_ctx = layers.Conv1D(16, kernel_size=1, activation='relu')(joint_ctx)  # project back to 16

        # ── FiLM: joint speed+accel context modulates sensor+gamerot ─────
        enc_conditioned = gated_film_layer_dropout(enc_combined, joint_ctx)             # (T/4, 96)

        # ── temporal refinement (no accel concat here anymore) ───────────
        x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enc_conditioned)
        x = layers.BatchNormalization()(x)

        # ── multi-head attention ────────────────────────────────────────
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x    = layers.Add()([x, attn])
        x    = layers.LayerNormalization()(x)

        # ── bidirectional LSTM stack ────────────────────────────────────
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
        )(x)

        # ── late fusion: device state ───────────────────────────────────
        enc_dev_flat = layers.GlobalAveragePooling1D()(enc_dev)
        x = layers.concatenate([x, enc_dev_flat])

        # ── classification head ─────────────────────────────────────────
        x   = layers.Dense(32, activation='relu')(x)
        x   = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

        return Model(
            inputs  = [inp_sensor, inp_gamerot, inp_speed, inp_accel, inp_device],
            outputs = out,
        )      


def build_model_tflite(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),    name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),    name="device_data")

    enc_sens = sensor_encoder(inp_sensor)
    enc_pos  = position_encoder(inp_position)
    enc_dev  = device_encoder(inp_device)

    x = layers.concatenate([enc_sens, enc_pos, enc_dev])
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)

    # BN frozen for TFLite compatibility
    x = layers.BatchNormalization(trainable=False)(x)

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # recurrent_dropout removed
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2)
    )(x)

    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    model = Model(
        inputs=[inp_sensor, inp_position, inp_device],
        outputs=out,
    )
    return model

def build_model_all(n_timesteps, n_sensor_feat, n_pos_feat, n_dev_feat, n_act_feat):
    inp_sensor   = Input(shape=(n_timesteps, n_sensor_feat), name="sensor_data")
    inp_position = Input(shape=(n_timesteps, n_pos_feat),    name="position_data")
    inp_device   = Input(shape=(n_timesteps, n_dev_feat),    name="device_data") 
    inp_activity = Input(shape=(n_act_feat,),                name="activity_data")
    
    enc_sens = sensor_encoder(inp_sensor)      
    enc_pos  = position_encoder(inp_position)  
    enc_dev  = device_encoder(inp_device) 

    # ── merge ─────────────────────────────────────────────────
    x = layers.concatenate([enc_sens, enc_pos, enc_dev])

    # ── temporal refinement ───────────────────────────────────
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # ── multi-head attention ───────────────────────────────────
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)

    # ── bidirectional LSTM stack ───────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    x = layers.concatenate([x, inp_activity])

    # ── classification head ───────────────────────────────────
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', name='distraction', dtype='float32')(x)

    model = Model(
        inputs  = [inp_sensor, inp_position, inp_device, inp_activity],
        outputs = out,
    )
    return model