import torch
from torch import nn
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from config import BINS, model_config


class MotionModel(nn.Module):
    """NeMo causal Conformer + known starting position + four categorical heads."""
    def __init__(self, bins=None, settings=None):
        super().__init__()
        self.settings = dict(model_config() if settings is None else settings)
        c = self.settings
        self.feat_in = c['feat_in']
        self.position_scale = c['position_scale']
        bins = BINS if bins is None else bins
        for j, b in enumerate(bins):
            self.register_buffer(f'bins_{j}', torch.as_tensor(b, dtype=torch.float32))
        # No subsampling: one encoded output for every available feature frame.
        # Two constant feature channels condition ALL layers on the known p(0).
        self.encoder = ConformerEncoder(
            feat_in=self.feat_in + 2, n_layers=c['layers'], d_model=c['hidden'],
            subsampling_factor=1, subsampling=None,
            self_attention_model='rel_pos', n_heads=c['attention_heads'],
            att_context_size=[c['left_context'], 0], att_context_style='regular',
            conv_kernel_size=c['conv_kernel'], conv_context_size='causal',
            conv_norm_type='layer_norm', dropout=c['dropout'],
            dropout_pre_encoder=c['dropout'], dropout_emb=0.0,
            dropout_att=c['dropout'], sync_max_audio_length=False,
        )
        # Linear pre_encode has no automatic audio-buffer chunk schedule in NeMo.
        # We call encoder.forward directly on nonoverlapping feature chunks and
        # schedule 10 Hz outputs ourselves. No trimming/overlap/lookahead is needed.
        self.encoder.streaming_cfg.cache_drop_size = 0
        self.encoder.streaming_cfg.drop_extra_pre_encoded = 0
        self.heads = nn.ModuleList(nn.Linear(c['hidden'], len(b)) for b in bins)

    @property
    def bins(self):
        return [getattr(self, f'bins_{j}') for j in range(4)]

    def forward(self, x, p0, cache=None):
        """x:[B,T,F] log1p magnitudes, p0:[B,2] metres; cache optional 3-tuple."""
        if x.ndim != 3 or x.shape[-1] != self.feat_in or x.shape[1] == 0:
            raise ValueError(f'Expected nonempty [B,T,{self.feat_in}] input')
        if p0.shape != (x.shape[0], 2):
            raise ValueError('p0 must have shape [batch, 2]')
        context = (p0.to(x) / self.position_scale).unsqueeze(1).expand(-1, x.shape[1], -1)
        features = torch.cat((x, context), dim=-1).transpose(1, 2)
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
        kwargs = {} if cache is None else dict(
            cache_last_channel=cache[0], cache_last_time=cache[1],
            cache_last_channel_len=cache[2])
        out = self.encoder(audio_signal=features, length=lengths, **kwargs)
        z = out[0].transpose(1, 2)
        next_cache = None if cache is None else (out[2], out[3], out[4])
        return [head(z) for head in self.heads], next_cache

    def initial_cache(self, batch_size=1):
        p = next(self.parameters())
        return self.encoder.get_initial_cache_state(batch_size=batch_size,
                                                    device=p.device, dtype=p.dtype)

    def predict_chunk(self, x, p0, cache=None):
        """Initialize caches even for the FIRST streaming chunk; returns updated caches."""
        if self.training:
            raise RuntimeError('Call model.eval() before streaming inference')
        if cache is None:
            cache = self.initial_cache(x.shape[0])
        with torch.inference_mode():
            return self.forward(x, p0, cache)
