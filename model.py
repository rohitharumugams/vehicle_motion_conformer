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


class MotionModel(nn.Module):
    """Optional frequency CNN + causal Conformer + four categorical heads."""

    def __init__(self, bins=None, settings=None):
        super().__init__()

        if settings is None:
            # New training runs use the CNN frontend.
            self.settings = dict(model_config())
            self.settings.setdefault("frontend", "frequency_cnn")
        else:
            # Old checkpoints without this field retain the original model.
            self.settings = dict(settings)
            self.settings.setdefault("frontend", "linear")

        c = self.settings
        self.feat_in = c["feat_in"]
        self.position_scale = c["position_scale"]

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
            encoder_feat_in = c["hidden"] + 2
        elif c["frontend"] == "linear":
            self.frontend = nn.Identity()
            encoder_feat_in = self.feat_in + 2
        else:
            raise ValueError(f"Unknown frontend: {c['frontend']}")

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

        return [head(z) for head in self.heads], next_cache

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
            return self.forward(x, p0, cache)
