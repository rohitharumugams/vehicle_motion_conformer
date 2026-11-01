import torch
from torch import nn
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

from config import BINS, model_config


class FrequencyCNN(nn.Module):
    """Process each spectrogram frame independently across frequency."""

    def __init__(self, n_freq=1025, embedding_dim=128, dropout=0.1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        reduced_freq = (n_freq + 1) // 2
        reduced_freq = (reduced_freq + 1) // 2

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16 * reduced_freq, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, time, frequencies = x.shape
        spectra = x.reshape(batch * time, 1, frequencies)
        embeddings = self.projection(self.convs(spectra))
        return embeddings.reshape(batch, time, -1)


class SpectroTemporalCNN(nn.Module):
    """2D spectro-temporal convolutions capturing Doppler rate of change across time and frequency."""

    def __init__(self, n_freq=1025, embedding_dim=128, dropout=0.1):
        super().__init__()
        import torch.nn.functional as F
        self._pad = F.pad
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 5), stride=(1, 2), padding=0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 5), stride=(1, 2), padding=0)
        self.act2 = nn.GELU()

        reduced_freq = (n_freq + 2 * 2 - 5) // 2 + 1
        reduced_freq = (reduced_freq + 2 * 2 - 5) // 2 + 1

        self.projection = nn.Sequential(
            nn.Linear(16 * reduced_freq, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Causal left padding in time: left=2, right=0; symmetric padding in frequency: left=2, right=2
        out = self._pad(x.unsqueeze(1), (2, 2, 2, 0))
        out = self.act1(self.conv1(out))
        out = self._pad(out, (2, 2, 2, 0))
        out = self.act2(self.conv2(out))
        B, C, T, F_red = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, T, C * F_red)
        return self.projection(out)


class MotionModel(nn.Module):
    """Optional frequency CNN + causal Conformer + four categorical heads."""

    def __init__(self, bins=None, settings=None):
        super().__init__()

        if settings is None:
            # New training runs use the CNN frontend.
            self.settings = dict(model_config())
            self.settings.setdefault("frontend", "frequency_cnn")
            self.settings.setdefault("spatial_conditioning", "concat")
        else:
            # Old checkpoints without this field retain the original model.
            self.settings = dict(settings)
            self.settings.setdefault("frontend", "linear")
            self.settings.setdefault("spatial_conditioning", "concat")

        c = self.settings
        self.feat_in = c["feat_in"]
        self.position_scale = c["position_scale"]
        self.spatial_conditioning = c["spatial_conditioning"]

        bins = BINS if bins is None else bins
        for j, b in enumerate(bins):
            self.register_buffer(
                f"bins_{j}",
                torch.as_tensor(b, dtype=torch.float32),
            )

        if c["frontend"] == "frequency_cnn":
            self.frontend = FrequencyCNN(
                n_freq=self.feat_in,
                embedding_dim=c["hidden"],
                dropout=c["dropout"],
            )
            raw_feat_dim = c["hidden"]
        elif c["frontend"] == "spectro_temporal_cnn":
            self.frontend = SpectroTemporalCNN(
                n_freq=self.feat_in,
                embedding_dim=c["hidden"],
                dropout=c["dropout"],
            )
            raw_feat_dim = c["hidden"]
        elif c["frontend"] == "linear":
            self.frontend = nn.Identity()
            raw_feat_dim = self.feat_in
        else:
            raise ValueError(f"Unknown frontend: {c['frontend']}")

        if self.spatial_conditioning == "film":
            self.film_gen = nn.Sequential(
                nn.Linear(2, 64),
                nn.GELU(),
                nn.Linear(64, 2 * raw_feat_dim),
            )
            encoder_feat_in = raw_feat_dim
        else:
            self.film_gen = None
            encoder_feat_in = raw_feat_dim + 2

        self.encoder = ConformerEncoder(
            feat_in=encoder_feat_in,
            n_layers=c["layers"],
            d_model=c["hidden"],
            subsampling_factor=1,
            subsampling=None,
            self_attention_model="rel_pos",
            n_heads=c["attention_heads"],
            att_context_size=[c["left_context"], 0],
            att_context_style="regular",
            conv_kernel_size=c["conv_kernel"],
            conv_context_size="causal",
            conv_norm_type="layer_norm",
            dropout=c["dropout"],
            dropout_pre_encoder=c["dropout"],
            dropout_emb=0.0,
            dropout_att=c["dropout"],
            sync_max_audio_length=False,
        )

        self.encoder.streaming_cfg.cache_drop_size = 0
        self.encoder.streaming_cfg.drop_extra_pre_encoded = 0

        self.heads = nn.ModuleList(
            nn.Linear(c["hidden"], len(b)) for b in bins
        )

        # Auxiliary direct position regression head: predicts (px, py) in metres.
        # Used only during training as an extra anchor loss; not used during decoding.
        self.position_head = nn.Sequential(
            nn.Linear(c["hidden"], 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    @property
    def bins(self):
        return [getattr(self, f"bins_{j}") for j in range(4)]

    def forward(self, x, p0, cache=None):
        if (
            x.ndim != 3
            or x.shape[-1] != self.feat_in
            or x.shape[1] == 0
        ):
            raise ValueError(
                f"Expected nonempty [B,T,{self.feat_in}] input"
            )

        if p0.shape != (x.shape[0], 2):
            raise ValueError("p0 must have shape [batch, 2]")

        spectral_features = self.frontend(x)

        if self.spatial_conditioning == "film":
            norm_p0 = p0.to(x) / self.position_scale
            film_weights = self.film_gen(norm_p0)
            gamma, beta = torch.chunk(film_weights, 2, dim=-1)
            conditioned_features = (1.0 + gamma.unsqueeze(1)) * spectral_features + beta.unsqueeze(1)
            features = conditioned_features.transpose(1, 2)
        else:
            context = (
                (p0.to(x) / self.position_scale)
                .unsqueeze(1)
                .expand(-1, x.shape[1], -1)
            )
            features = torch.cat(
                (spectral_features, context), dim=-1
            ).transpose(1, 2)

        lengths = torch.full(
            (x.shape[0],),
            x.shape[1],
            dtype=torch.long,
            device=x.device,
        )

        kwargs = {}
        if cache is not None:
            kwargs = {
                "cache_last_channel": cache[0],
                "cache_last_time": cache[1],
                "cache_last_channel_len": cache[2],
            }

        out = self.encoder(
            audio_signal=features,
            length=lengths,
            **kwargs,
        )

        z = out[0].transpose(1, 2)
        next_cache = (
            None if cache is None else (out[2], out[3], out[4])
        )

        return [head(z) for head in self.heads], next_cache, self.position_head(z)

    def initial_cache(self, batch_size=1):
        p = next(self.parameters())
        return self.encoder.get_initial_cache_state(
            batch_size=batch_size,
            device=p.device,
            dtype=p.dtype,
        )

    def predict_chunk(self, x, p0, cache=None):
        if self.training:
            raise RuntimeError("Call model.eval() before streaming inference")

        if cache is None:
            cache = self.initial_cache(x.shape[0])

        with torch.inference_mode():
            logits, next_cache, _ = self.forward(x, p0, cache)
            return logits, next_cache
