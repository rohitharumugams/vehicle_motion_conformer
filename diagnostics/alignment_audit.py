"""Verify exporter geometry, WAV/STFT pairing, and acoustic delay without modifying data."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config as C
sys.path.insert(0,'/Users/rohith/Desktop/Rohith/Doppler/DopplerSim_2.0')
from doppler_sim.path2d.synthesis import resample_path_constant_speed,solve_retarded_time_path
from doppler_sim.batch.phase1_state import _canonicalize_state_2d
import soundfile as sf
import librosa
rows=[];delays=[];gaps=[];stateerrs=[]
for f in sorted(Path(C.DATA).rglob(C.STATE)):
 meta=f.parent;p=meta.parent;m=json.loads((meta/'simulation_parameters.json').read_text());s=np.load(meta/'state_frames.npy');can=np.load(f);t=np.load(meta/'frame_times.npy').astype(float);wav=next(p.glob('*.wav'));audio,sr=sf.read(wav);cfg=m['render_path2d'];traj=resample_path_constant_speed(np.array(m['path_xy']),speed_mps=cfg['v2_kmph']/3.6,sr=sr)
 expect=np.stack([np.interp(t,traj['t'],traj[k]) for k in ['x','vx','y','vy']],1);expect[:,[0,2]]-=m['mic_xy']
 canon=_canonicalize_state_2d(expect);r=np.linalg.norm(s[:,[0,2]],axis=1);tr,rr=solve_retarded_time_path(t,t,s[:,0],s[:,2]);valid=np.isfinite(tr);delays.extend(t[valid]-tr[valid]);
 # Acoustic frame-center age relative to scheduled current-state output.
 seconds=np.arange(1,int(np.floor(t[-1]*10+1e-6))+1)/10;idx=np.searchsorted(t+1024/22050,seconds+1e-8,side='right')-1;validout=(idx>=0)&np.isfinite(tr[np.maximum(idx,0)]);gaps.extend(seconds[validout]-tr[idx[validout]])
 yy=librosa.resample(audio.astype(np.float32),orig_sr=sr,target_sr=22050);spec=abs(librosa.stft(yy,n_fft=2048,hop_length=256,window='hann',center=True));saved=np.load(p/'spectrograms/stft.npy');
 row={'sample':p.name,'sample_rate':sr,'wav_samples':len(audio),'trajectory_samples':len(traj['t']),'state_max_abs_difference':float(abs(expect-s).max()),'canonical_max_abs_difference':float(abs(canon-can).max()),'frame_time_max_abs_difference_s':float(abs(t-np.arange(len(t))*256/22050).max()),'stft_relative_l2_error':float(np.linalg.norm(spec-saved)/max(np.linalg.norm(saved),1e-12)),'stft_max_abs_difference':float(abs(spec-saved).max()),'propagation_delay_median_s':float(np.median(t[valid]-tr[valid])),'num_emitters':cfg['num_emitters']};rows.append(row)
 if len(rows)%20==0:print('audited',len(rows),flush=True)
summary={'clips':len(rows),'state_max_abs_difference':max(r['state_max_abs_difference'] for r in rows),'canonical_max_abs_difference':max(r['canonical_max_abs_difference'] for r in rows),'frame_time_max_abs_difference_s':max(r['frame_time_max_abs_difference_s'] for r in rows),'stft_relative_l2_error_median_max':np.percentile([r['stft_relative_l2_error'] for r in rows],[50,100]).tolist(),'sample_count_mismatches':sum(r['wav_samples']!=r['trajectory_samples'] for r in rows),'propagation_delay_seconds_p0_p50_p95_p100':np.percentile(delays,[0,50,95,100]).tolist(),'latest_frame_center_emission_age_seconds_p0_p50_p95_p100':np.percentile(gaps,[0,50,95,100]).tolist()}
Path('diagnostics/alignment_audit.json').write_text(json.dumps({'summary':summary,'clips':rows},indent=2));print(json.dumps(summary,indent=2),flush=True)
