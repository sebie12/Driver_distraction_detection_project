POSITION_FEATURES_EXTRA = [
    ("LINACC",  "v1",  "LINACC_v1"),  ("LINACC",  "v2",  "LINACC_v2"), ("LINACC",  "v3",  "LINACC_v3"),
    ("GYRO",    "v1",  "GYRO_v1"),    ("GYRO",    "v2",  "GYRO_v2"),   ("GYRO",    "v3",  "GYRO_v3"),
    ("GRAV",    "v1",  "GRAV_v1"),    ("GRAV",    "v2",  "GRAV_v2"),
    ("GRAV",    "v3",  "GRAV_v3"),    ("GRAV",    "mag", "GRAV_mag"),
    ("ROT",     "v1",  "ROT_v1"),     ("ROT",     "v2",  "ROT_v2"),
    ("ROT",     "v3",  "ROT_v3"),     ("ROT",     "mag", "ROT_mag"),
    ("GAMEROT", "v1",  "GAMEROT_v1"), ("GAMEROT", "v2",  "GAMEROT_v2"),
    ("GAMEROT", "v3",  "GAMEROT_v3"), ("GAMEROT", "mag", "GAMEROT_mag"),
]

class ModelConfig:
    def __init__(self, version, use_extra_features=False, include_accel_flags=False, include_instant_accel=True):
        self.version = version
        self.use_extra_features = use_extra_features
        self.include_accel_flags = include_accel_flags
        self.include_instant_accel = include_instant_accel

    def get_streams(self):
        streams = {}
        
        parts = self.version.split('.')
        base_version = parts[0]
        # Normalize: ensure "v" prefix for version matching
        if not base_version.startswith("v"):
            base_version = "v" + base_version
        suffix = parts[1] if len(parts) > 1 else '0'
        
        # --- SENSOR ---
        sensor = []
        if suffix == '0':
            sensor.extend([
                ("GYRO", "mag", "GYRO_mag"), 
                ("LINACC", "mag", "LINACC_mag")
            ])
        elif suffix == '1':
            sensor.extend([
                ("GYRO", "mag", "GYRO_mag"), ("GYRO", "v1", "GYRO_v1"), ("GYRO", "v2", "GYRO_v2"), ("GYRO", "v3", "GYRO_v3"),
                ("LINACC", "mag", "LINACC_mag"), ("LINACC", "v1", "LINACC_v1"), ("LINACC", "v2", "LINACC_v2"), ("LINACC", "v3", "LINACC_v3")
            ])
        elif suffix == '2':
            sensor.extend([
                ("GYRO", "mag", "GYRO_mag"), ("GYRO", "v1", "GYRO_v1"), ("GYRO", "v2", "GYRO_v2"), ("GYRO", "v3", "GYRO_v3"),
                ("LINACC", "mag", "LINACC_mag")
            ])
        elif suffix == '3':
            sensor.extend([
                ("GYRO", "mag", "GYRO_mag"), ("GYRO", "v1", "GYRO_v1"), ("GYRO", "v2", "GYRO_v2"), ("GYRO", "v3", "GYRO_v3"),
                ("LINACC", "mag", "LINACC_mag"), ("GAMEROT", "angular_distance", "GAMEROT_angular_distance")
            ])
        elif suffix in ('4', '5'):
            sensor.extend([
                ("GYRO", "mag", "GYRO_mag"), ("GYRO", "v1", "GYRO_v1"), ("GYRO", "v2", "GYRO_v2"), ("GYRO", "v3", "GYRO_v3"),
                ("LINACC", "mag", "LINACC_mag"),
                ("GAMEROT", "angular_distance", "GAMEROT_angular_distance"),
                ("GRAV", "reorientation", "GRAV_reorientation")
            ])
            if suffix == '5':
                # .5 = .4 motion block + dense, magnitude-orthogonal spectral-texture
                # channels. A mounted phone transmits road vibration (high-freq); a
                # hand-held phone damps it (low-freq). Computed on the RAW signal
                # (ignores denoise) so the discriminative high-freq content survives.
                sensor.extend([
                    ("GYRO",   "hf_ratio", "GYRO_hf_ratio"),
                    ("GYRO",   "zcr",      "GYRO_zcr"),
                    ("LINACC", "hf_ratio", "LINACC_hf_ratio"),
                    ("LINACC", "zcr",      "LINACC_zcr"),
                ])
        else:
            raise ValueError(f"Unknown suffix .{suffix} in version {self.version}")

        streams["sensor"] = sensor
        
        # --- SPEED / POSITION ---
        if base_version in ["v4", "v5", "v6", "v7", "v8", "v12", "v13", "v14", "all"]:
            speed = [("LOC", "speed_mps", "speed_mps")]
            if self.use_extra_features:
                speed.extend(POSITION_FEATURES_EXTRA)
            streams["speed"] = speed
        else:
            streams["speed"] = []
            
        streams["position"] = []
        
        # --- GAMEROT ---
        if base_version in ["v7", "v8", "v9", "v10", "v11", "v13", "all"]:
            streams["gamerot"] = [("GAMEROT", "angular_distance", "GAMEROT_angular_distance")]
        else:
            streams["gamerot"] = []

        # --- ACCEL ---
        if base_version in ["v8", "v9", "v10", "v11", "v12", "v13", "all"]:
            accel = []
            if base_version != "v10":
                if self.include_instant_accel:
                    accel.append(("LOC", "instant_acceleration", "instant_acceleration"))
                
                # Only include peak_acceleration if include_accel_flags is True
                if self.include_accel_flags:
                    accel.append(("LOC", "peak_acceleration", "peak_acceleration"))
            
            if self.include_accel_flags:
                accel.extend([
                    ("LOC", "hard_brake_flag", "hard_brake_flag"),
                    ("LOC", "hard_accel_flag", "hard_accel_flag"),
                ])
            streams["accel"] = accel
        else:
            streams["accel"] = []

        # --- DEVICE (Always present) ---
        # ds_handsfree (audio routed to Bluetooth/headset) replaces ds_audio_active:
        # audio_active also fires on music / GPS voice (not distractions), whereas
        # hands-free routing is a cleaner proxy for a non-handheld call.
        streams["device"] = [
            ("DEVICE", "ds_device_locked", "ds_device_locked"),
            ("DEVICE", "ds_handsfree", "ds_handsfree"),
        ]

        return streams
