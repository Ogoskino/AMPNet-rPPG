#!/usr/bin/env python3
"""Prepare raw iBVP ZIP archives or extracted frame folders for AMPNet.

Requires 64-bit Python 3.12.x. First run: python prepare_ibvp.py --setup
Setup installs pinned packages and the face model in an isolated environment.

Example:
  python prepare_ibvp.py --input-dir /data/iBVP_Dataset --output-dir ./datasets \
      --face-model /models/blaze_face_short_range.tflite

Defaults retain the paper's 5376 selected frames and 128-frame clip boundaries.
Each recording is packed into three chronological 1792-frame rows for the
current AMPNet loader; no selected frames are discarded. See PREPROCESSING.md
for paper/code ambiguities, normalization scope, timing, and split provenance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import zipfile


# Standalone setup uses only the standard library. It runs before NumPy imports.
import platform
import struct
import subprocess
import tempfile
import urllib.request
import venv

SETUP_SCHEMA = 1
SETUP_PINS = ['numpy==1.26.4','opencv-contrib-python==4.11.0.86','mediapipe==0.10.32']
TORCH_PIN = 'torch==2.6.0'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
MODEL_SHA256 = 'b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f'
MODEL_NAME = 'blaze_face_short_range.tflite'
SETUP_RECIPE = hashlib.sha256(json.dumps([SETUP_PINS,TORCH_PIN,MODEL_URL,MODEL_SHA256,'Python3.12-64bit']).encode()).hexdigest()


def check_setup_python(version=None, bits=None):
    version = sys.version_info if version is None else version
    bits = struct.calcsize('P')*8 if bits is None else bits
    if tuple(version[:2]) != (3,12) or bits != 64:
        raise ValueError('Use 64-bit Python 3.12.x. On Windows: py -3.12 prepare_ibvp.py --setup. Python itself must be installed before this script can run.')


def env_python(env_dir):
    return Path(env_dir)/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def clean_child_env():
    result=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PIP_TARGET','PIP_PREFIX','PIP_USER'):
        result.pop(key,None)
    result['PYTHONNOUSERSITE']='1'
    return result


def setup_pip_commands(python_path, platform_name=None):
    system=platform.system() if platform_name is None else platform_name
    base=[str(python_path),'-I','-m','pip','--isolated','--require-virtualenv',
          'install','--disable-pip-version-check','--no-input','--only-binary=:all:']
    torch_index='https://pypi.org/simple' if system=='Darwin' else 'https://download.pytorch.org/whl/cpu'
    return [base+['--index-url',torch_index,TORCH_PIN],
            base+['--index-url','https://pypi.org/simple']+SETUP_PINS]


def validate_env_target(env_dir, script_path=None):
    target=Path(env_dir).expanduser().resolve()
    script=Path(__file__ if script_path is None else script_path).resolve()
    if target in (Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),script.parent) or target in script.parents:
        raise ValueError('Choose a dedicated environment subdirectory, not Python, the script folder or an ancestor')
    if target.exists():
        if not target.is_dir(): raise ValueError('Environment path must be a directory')
        if any(target.iterdir()):
            marker=target/'.ibvp-managed.json'
            try: owned=json.loads(marker.read_text(encoding='utf-8'))
            except (OSError,ValueError): raise ValueError('Environment directory is nonempty and is not owned by this converter; choose a new --env-dir') from None
            if not isinstance(owned,dict) or owned.get('managed_by')!='prepare_ibvp' or owned.get('schema')!=SETUP_SCHEMA:
                raise ValueError('Unrecognized environment ownership marker; choose a new --env-dir')
    return target


def strip_setup_args(argv):
    result=[];i=0
    while i<len(argv):
        if argv[i]=='--setup': i+=1;continue
        if argv[i]=='--env-dir':
            if i+1>=len(argv): raise ValueError('--env-dir requires a path')
            i+=2;continue
        if argv[i].startswith('--env-dir='): i+=1;continue
        result.append(argv[i]);i+=1
    return result


def download_face_model(destination):
    destination=Path(destination)
    if destination.exists():
        actual=hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual!=MODEL_SHA256: raise ValueError(f'Existing face model failed SHA256 verification: {destination}')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='face-download-',suffix='.part',dir=destination.parent,delete=False) as f:
            temporary=Path(f.name)
            with urllib.request.urlopen(MODEL_URL,timeout=60) as response:
                length=0
                while True:
                    chunk=response.read(65536)
                    if not chunk: break
                    length+=len(chunk)
                    if length>1024*1024: raise ValueError('Unexpected face-model download size')
                    f.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise ValueError('Downloaded face model failed SHA256 verification')
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def probe_setup(python_path, model_path):
    code=r"""
import sys,json,platform,importlib.metadata as md
import numpy as np, torch, cv2, mediapipe as mp
assert sys.version_info[:2]==(3,12)
assert sys.prefix!=sys.base_prefix
assert np.__version__=='1.26.4'
assert torch.__version__.split('+')[0]=='2.6.0'
assert md.version('opencv-contrib-python')=='4.11.0.86'
assert mp.__version__=='0.10.32'
assert np.array_equal(torch.from_numpy(np.arange(4,dtype=np.float32)).numpy(),np.arange(4,dtype=np.float32))
options=mp.tasks.vision.FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path=sys.argv[1]),running_mode=mp.tasks.vision.RunningMode.IMAGE)
with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
    detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.zeros((64,64,3),dtype=np.uint8)))
print(json.dumps({'status':'passed','python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'numpy':np.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'mediapipe':mp.__version__,'resolved_packages':{d.metadata['Name']:d.version for d in md.distributions()}}))
"""
    result=subprocess.run([str(python_path),'-I','-c',code,str(model_path)],env=clean_child_env(),text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(f'Environment verification failed:\n{result.stderr[-6000:]}')
    try: return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError): raise RuntimeError('Environment probe did not return a valid receipt') from None


def install_setup(env_dir):
    check_setup_python()
    system,machine=platform.system(),platform.machine().lower()
    if not ((system in ('Windows','Linux') and machine in ('amd64','x86_64')) or
            (system=='Darwin' and machine in ('arm64','aarch64'))):
        raise ValueError('Pinned MediaPipe setup supports Windows/Linux x86-64 and Apple Silicon macOS. Use a supported 64-bit Python 3.12 installation.')
    target=validate_env_target(env_dir)
    marker_path=target/'.ibvp-managed.json'
    marker=json.loads(marker_path.read_text(encoding='utf-8')) if marker_path.exists() else {}
    py=env_python(target)
    model=target/'models'/MODEL_NAME
    if marker.get('status')=='ready' and marker.get('recipe')==SETUP_RECIPE and py.is_file():
        model=download_face_model(model)
        probe=probe_setup(py,model)
        print(f'Existing setup verified: {target}',flush=True)
        return target
    target.mkdir(parents=True,exist_ok=True)
    marker={'managed_by':'prepare_ibvp','schema':SETUP_SCHEMA,'status':'installing','recipe':SETUP_RECIPE,
            'creator_python':sys.version,'target':str(target)}
    marker_path.write_text(json.dumps(marker,indent=2)+'\n',encoding='utf-8')
    try:
        print(f'Creating isolated Python 3.12 environment: {target}',flush=True)
        venv.EnvBuilder(with_pip=True,system_site_packages=False,clear=False).create(target)
        for command in setup_pip_commands(py):
            print('Installing pinned dependencies...',flush=True)
            subprocess.run(command,check=True,env=clean_child_env())
        subprocess.run([str(py),'-I','-m','pip','--isolated','--require-virtualenv','check'],check=True,env=clean_child_env())
        print('Downloading/checking the MediaPipe face model...',flush=True)
        model=download_face_model(model)
        probe=probe_setup(py,model)
        marker.update(status='ready',verification=probe,face_model_sha256=MODEL_SHA256,face_model=str(model))
        marker_path.write_text(json.dumps(marker,indent=2)+'\n',encoding='utf-8')
        print(f'Setup and dependency checks passed: {target}',flush=True)
        return target
    except Exception as exc:
        marker.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        marker_path.write_text(json.dumps(marker,indent=2)+'\n',encoding='utf-8')
        raise


def standalone_bootstrap(argv):
    quick=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    quick.add_argument('--setup',action='store_true')
    quick.add_argument('--env-dir',type=Path,default=Path(__file__).resolve().parent/'.ibvp-venv')
    quick.add_argument('--use-current-env',action='store_true')
    options,remaining=quick.parse_known_args(argv)
    if '--help' in argv or '-h' in argv: return None
    if options.setup and options.use_current_env: raise ValueError('--setup and --use-current-env cannot be combined')
    target=options.env_dir.expanduser().resolve()
    for flag in ('--input-dir','--output-dir'):
        scan=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
        scan.add_argument(flag,type=Path)
        parsed,_=scan.parse_known_args(argv)
        requested=getattr(parsed,flag[2:].replace('-','_'))
        if requested is not None:
            folder=requested.expanduser().resolve()
            if target==folder or folder in target.parents:
                raise ValueError('--env-dir must be outside the raw-input and processed-output directories')
    if options.setup:
        target=install_setup(target)
        if not remaining:
            print('Run this same file with --input-dir and --output-dir to process data; the managed environment and face model will be used automatically.',flush=True)
            return 0
    if options.use_current_env: return None
    marker_path=target/'.ibvp-managed.json'
    if not marker_path.exists():
        if '--env-dir' in argv or any(x.startswith('--env-dir=') for x in argv):
            raise ValueError('No managed environment at --env-dir. Run with --setup first.')
        return None
    marker=json.loads(marker_path.read_text(encoding='utf-8'))
    if not isinstance(marker,dict) or marker.get('managed_by')!='prepare_ibvp' or marker.get('schema')!=SETUP_SCHEMA:
        raise ValueError('Invalid managed environment marker')
    if marker.get('status')!='ready' or marker.get('recipe')!=SETUP_RECIPE:
        raise ValueError('Environment setup is incomplete or its recipe changed. Run --setup again.')
    py=env_python(target)
    if not py.is_file(): raise ValueError('Managed Python is missing. Run --setup again.')
    if Path(sys.prefix).resolve()==target: return None
    command=[str(py),'-I',str(Path(__file__).resolve())]+strip_setup_args(argv)+['--env-dir',str(target)]
    return subprocess.run(command,env=clean_child_env()).returncode


if __name__=='__main__':
    try:
        _bootstrap_status=standalone_bootstrap(sys.argv[1:])
        if _bootstrap_status is not None: raise SystemExit(_bootstrap_status)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as _setup_error:
        print(f'ERROR: {_setup_error}',file=sys.stderr)
        raise SystemExit(1)

try:
    import numpy as np
except ImportError:
    np = None


PAPER_TRAIN = ['p02','p03','p04','p06','p11','p12','p13','p14','p15',
               'p17','p18','p21','p23','p24','p25','p27','p30','p32']
PAPER_TEST = ['p22','p26','p28','p05']  # Asian, Black, Caucasian, Mixed
VERSION = '1.1.0'


def natural_key(value):
    return tuple((1,int(p)) if p.isdigit() else (0,p.lower())
                 for p in re.split(r'(\d+)', str(value)))


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class FrameSource:
    """Read selected frames without extracting a ZIP or changing raw files."""
    def __init__(self, path: Path, modality: str):
        self.path = Path(path)
        if modality not in ('rgb','thermal'):
            raise ValueError('modality must be rgb or thermal')
        extensions = {'.bmp','.png','.jpg','.jpeg'} if modality == 'rgb' else {'.raw'}
        self.archive = None
        if self.path.is_dir():
            self.names = sorted([str(p.relative_to(self.path)) for p in self.path.rglob('*')
                                 if p.is_file() and p.suffix.lower() in extensions],
                                key=lambda n:natural_key(Path(n).name))
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            self.names = sorted([i.filename for i in self.archive.infolist()
                                 if not i.is_dir() and Path(i.filename).suffix.lower() in extensions
                                 and '__MACOSX' not in i.filename.split('/')],
                                key=lambda n:natural_key(Path(n).name))
        else:
            raise ValueError(f'Expected extracted folder or ZIP archive: {self.path}')
        if not self.names:
            self.close()
            raise ValueError(f'No {modality} frames found: {self.path}')
        stems = [Path(n).stem for n in self.names]
        if len(set(stems)) != len(stems):
            self.close()
            raise ValueError(f'Duplicate frame identifiers in {self.path}')
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self):
        if self.archive is not None: self.archive.close()
    def read(self, index):
        name = self.names[index]
        return self.archive.read(name) if self.archive else (self.path/name).read_bytes()
    def timestamps_ms(self):
        try: values = np.array([int(Path(n).stem) for n in self.names],dtype=np.int64)
        except ValueError: return None
        if values.min() < 10**11 or not np.all(np.diff(values)>0): return None
        return values


def sample_indices(length: int, target: int, mode: str, seed: int):
    if length <= 0 or target <= 0 or target > length:
        raise ValueError(f'Need at least {target} frames; found {length}')
    if mode == 'legacy-random':
        return np.array(sorted(random.Random(seed).sample(range(length),target)),dtype=np.int64)
    if mode == 'uniform':
        return np.rint(np.linspace(0,length-1,target)).astype(np.int64)
    raise ValueError(f'Unknown sampling mode: {mode}')


def read_bvp(path: Path, column: str = 'BVP'):
    with Path(path).open('r',newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f'{path}: expected named BVP column {column!r}')
        try: values = np.array([float(row[column]) for row in reader],dtype=np.float32)
        except (TypeError,ValueError,KeyError) as e: raise ValueError(f'Invalid BVP values in {path}') from e
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Empty or non-finite BVP signal in {path}')
    return values


def thermal_celsius(payload: bytes, width=640, height=512):
    if len(payload) != width*height*2:
        raise ValueError(f'Thermal frame has {len(payload)} bytes; expected {width*height*2}')
    return np.frombuffer(payload,dtype='<u2').reshape(height,width).astype(np.float32)*np.float32(.04)-np.float32(273.15)


def crop_resize(image: np.ndarray, box, size=64):
    import cv2
    shape = (size,size) + image.shape[2:]
    if box is None: return np.zeros(shape,dtype=image.dtype)
    x,y,w,h = box
    if not all(math.isfinite(v) for v in box) or w<=0 or h<=0:
        raise ValueError(f'Invalid face box: {box}')
    ih,iw = image.shape[:2]
    x0,y0 = max(0,int(x)),max(0,int(y))
    x1,y1 = min(iw,int(x+w)),min(ih,int(y+h))
    if x1<=x0 or y1<=y0: raise ValueError(f'Face box outside image: {box}')
    return cv2.resize(image[y0:y1,x0:x1],(size,size),interpolation=cv2.INTER_LINEAR)


def normalize_inplace(features, labels, stats):
    r,t,lo,hi = (float(stats[k]) for k in ('rgb_max','thermal_max','bvp_min','bvp_max'))
    if not all(math.isfinite(v) for v in (r,t,lo,hi)) or r<=0 or t<=0 or hi<=lo:
        raise ValueError('Cannot normalize zero/missing video or constant BVP; inspect face detection and source signal')
    features[..., :3] /= r
    features[..., 3] /= t
    labels -= lo
    labels /= hi-lo


def discover_sessions(root):
    sessions = []
    for p in Path(root).rglob('*_bvp.csv'):
        match = re.fullmatch(r'(p\d+)_[a-z]',p.stem[:-4],re.I)
        if match:
            sessions.append({'subject':match.group(1).lower(),'session':p.stem[:-4],
                             'path':str(p.parent.resolve()),'bvp':str(p.resolve()),
                             'confidential_no_media_use': any('confidential' in x.lower() for x in p.parts)})
    if not sessions: raise ValueError(f'No <session>_bvp.csv recordings under {root}')
    names = [s['session'] for s in sessions]
    if len(names)!=len(set(names)): raise ValueError('Duplicate session folders; choose one dataset root')
    return sessions


def resolve_split(sessions, train_subjects, test_subjects):
    if len(set(train_subjects))!=len(train_subjects) or len(set(test_subjects))!=len(test_subjects):
        raise ValueError('Duplicate participant IDs in split')
    if set(train_subjects)&set(test_subjects): raise ValueError('Train/test subjects overlap')
    known = {s['subject'] for s in sessions}
    missing = (set(train_subjects)|set(test_subjects))-known
    if missing: raise ValueError(f'Requested subjects missing from dataset: {sorted(missing)}')
    selected = []
    for split, subjects in [('train',train_subjects),('test',test_subjects)]:
        for subject in subjects:
            for s in sorted((s for s in sessions if s['subject']==subject),key=lambda s:natural_key(s['session'])):
                selected.append(dict(s,split=split))
    return selected


def stream_path(session, suffix):
    base = Path(session['path'])/(session['session']+suffix)
    if base.is_dir(): return base
    if base.with_suffix('.zip').is_file(): return base.with_suffix('.zip')
    raise ValueError(f'Missing stream folder/ZIP: {base}')


def nearest_indices(source_times, target_times):
    right = np.searchsorted(source_times,target_times).clip(0,len(source_times)-1)
    left = (right-1).clip(0,len(source_times)-1)
    return np.where(np.abs(source_times[left]-target_times)<=np.abs(source_times[right]-target_times),left,right)


class FaceLocator:
    def __init__(self, backend, model_path, confidence):
        import mediapipe as mp
        self.mp = mp
        self.backend = ('legacy' if hasattr(mp,'solutions') else 'tasks') if backend=='auto' else backend
        self.version = mp.__version__
        if self.backend=='legacy':
            if not hasattr(mp,'solutions'):
                raise ValueError('This MediaPipe has no legacy solutions API. Use --face-backend tasks with --face-model, or a separate mediapipe==0.10.21 environment.')
            self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=confidence)
        else:
            if model_path is None or not Path(model_path).is_file():
                raise ValueError('MediaPipe Tasks needs --face-model PATH to a compatible face detector .tflite; see PREPROCESSING.md')
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(model_path).resolve())),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,min_detection_confidence=confidence)
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    def locate(self, image):
        if self.backend=='legacy':
            result = self.detector.process(image)
            if not result.detections: return None
            b=result.detections[0].location_data.relative_bounding_box
            h,w=image.shape[:2]
            return int(b.xmin*w),int(b.ymin*h),int(b.width*w),int(b.height*h)
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(image)))
        if not result.detections: return None
        b=result.detections[0].bounding_box
        return b.origin_x,b.origin_y,b.width,b.height
    def close(self): self.detector.close()


def array_stats(features, labels):
    return {'rgb_max':float(np.max(features[...,:3])), 'thermal_max':float(np.max(features[...,3])),
            'bvp_min':float(np.min(labels)), 'bvp_max':float(np.max(labels))}


def process_session(session, args, locator, destination, label_destination, audit_dir):
    import cv2
    bvp = read_bvp(Path(session['bvp']),args.bvp_column)
    seed = args.seed + int(hashlib.sha256(session['session'].encode()).hexdigest()[:8],16)
    rgb_file,thermal_file=stream_path(session,'_rgb'),stream_path(session,'_t')
    with FrameSource(rgb_file,'rgb') as rgb, FrameSource(thermal_file,'thermal') as thermal:
        rt,tt=rgb.timestamps_ms(),thermal.timestamps_ms()
        if len(bvp)!=len(rgb.names):
            raise ValueError(f"{session['session']}: BVP length {len(bvp)} differs from RGB {len(rgb.names)}; explicit resampling is required")
        if args.alignment=='timestamp':
            if rt is None or tt is None: raise ValueError('Timestamp alignment requires millisecond timestamps in both frame filenames')
            valid=np.flatnonzero((rt>=tt[0]) & (rt<=tt[-1]))
            positions=sample_indices(len(valid),args.frames,args.sampling,seed)
            rgb_idx=valid[positions]
            thermal_idx=nearest_indices(tt,rt[rgb_idx])
            errors=np.abs(tt[thermal_idx]-rt[rgb_idx])
            if errors.max()>args.max_pair_offset_ms:
                raise ValueError(f'Thermal pairing exceeds --max-pair-offset-ms ({errors.max()} ms)')
        else:
            rgb_idx=sample_indices(min(len(rgb.names),len(thermal.names),len(bvp)),args.frames,args.sampling,seed)
            thermal_idx=rgb_idx.copy()
        label_destination[:]=bvp[rgb_idx]
        rgb_hash,thermal_hash=hashlib.sha256(),hashlib.sha256()
        missing={'rgb':0,'thermal':0}
        rows=[]
        started=time.perf_counter()
        for output_idx,(ri,ti) in enumerate(zip(rgb_idx,thermal_idx)):
            rp,tp=rgb.read(int(ri)),thermal.read(int(ti))
            rgb_hash.update(rp);thermal_hash.update(tp)
            decoded=cv2.imdecode(np.frombuffer(rp,dtype=np.uint8),cv2.IMREAD_COLOR)
            if decoded is None: raise ValueError(f'Cannot decode {rgb.names[ri]}')
            rgb_image=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
            temp=thermal_celsius(tp,args.thermal_width,args.thermal_height)
            thermal_view=cv2.cvtColor(cv2.applyColorMap(cv2.normalize(temp,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
            boxes=[locator.locate(rgb_image),locator.locate(thermal_view)]
            for modality,box in zip(('rgb','thermal'),boxes):
                if box is None:
                    missing[modality]+=1
                    if args.missing_face=='error': raise ValueError(f"No {modality} face at {session['session']} selected frame {output_idx}")
            destination[output_idx,...,:3]=crop_resize(rgb_image,boxes[0])
            destination[output_idx,...,3]=crop_resize(temp,boxes[1])
            rows.append({'selected_frame':output_idx,'rgb_index':int(ri),'thermal_index':int(ti),'bvp_index':int(ri),
                         'rgb_file':rgb.names[ri],'thermal_file':thermal.names[ti],
                         'rgb_time_ms':None if rt is None else int(rt[ri]),'thermal_time_ms':None if tt is None else int(tt[ti]),
                         'rgb_box':boxes[0],'thermal_box':boxes[1]})
            if (output_idx+1)%512==0 or output_idx+1==args.frames:
                print(f"  {session['session']}: {output_idx+1}/{args.frames} frames; missing faces {missing}",flush=True)
        if any(missing[m]/args.frames>args.max_missing_face_fraction for m in missing):
            raise ValueError(f"{session['session']}: missing face fraction exceeds {args.max_missing_face_fraction}: {missing}")
        stats=array_stats(destination,label_destination)
        if stats['rgb_max']<=0 or stats['thermal_max']<=0 or stats['bvp_max']<=stats['bvp_min']:
            raise ValueError(f"{session['session']}: entirely missing modality or constant BVP; cannot export a usable recording")
        if args.normalization=='recording': normalize_inplace(destination,label_destination,stats)
        audit_path=audit_dir/(session['session']+'_frames.json')
        audit_path.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
        offsets=None if rt is None or tt is None else tt[thermal_idx]-rt[rgb_idx]
        details={**session,'frames':args.frames,'rows':args.frames//args.row_frames,
                 'rgb_source_frames':len(rgb.names),'thermal_source_frames':len(thermal.names),'bvp_source_samples':len(bvp),
                 'sampling_seed':seed,'alignment':args.alignment,'missing_faces':missing,'raw_normalization_stats':stats,
                 'selected_rgb_payload_sha256':rgb_hash.hexdigest(),'selected_thermal_payload_sha256':thermal_hash.hexdigest(),
                 'bvp_sha256':sha256(session['bvp']),'frame_audit':str(audit_path.name),
                 'frame_audit_sha256':sha256(audit_path),
                 'pair_offset_ms_quantiles':None if offsets is None else np.quantile(offsets,[0,.25,.5,.75,1]).tolist(),
                 'selected_duration_seconds':None if rt is None else float((rt[rgb_idx[-1]]-rt[rgb_idx[0]])/1000),
                 'elapsed_seconds':time.perf_counter()-started}
        return details


def verify_output(output_dir):
    """Reload exported plain tensors safely and check every stored frame."""
    import torch
    output_dir=Path(output_dir)
    manifest=json.loads((output_dir/'manifest.json').read_text(encoding='utf-8'))
    results={}
    for split in ('train','test'):
        expected=manifest['outputs'][split]
        paths=[output_dir/expected[k] for k in ('features','labels')]
        for path in paths:
            if sha256(path)!=expected['sha256'][path.name]: raise ValueError(f'Checksum mismatch: {path}')
        features=torch.load(paths[0],map_location='cpu',weights_only=True,mmap=True)
        labels=torch.load(paths[1],map_location='cpu',weights_only=True,mmap=True)
        if list(features.shape)!=expected['feature_shape'] or list(labels.shape)!=expected['label_shape']:
            raise ValueError(f'Unexpected saved {split} tensor shapes')
        if features.dtype!=torch.float32 or labels.dtype!=torch.float32: raise ValueError('Expected float32 tensors')
        if features.ndim!=5 or features.shape[-3:]!=(64,64,4) or labels.shape!=features.shape[:2]:
            raise ValueError('Invalid channel-last feature/BVP layout')
        if features.shape[1]%128: raise ValueError('Rows must contain complete128-frame clips')
        minimum,maximum=float('inf'),float('-inf')
        for row in range(len(features)):
            for start in range(0,features.shape[1],128):
                piece=features[row,start:start+128]
                if not torch.isfinite(piece).all(): raise ValueError('Non-finite video values')
                minimum=min(minimum,float(piece.min()));maximum=max(maximum,float(piece.max()))
                if float(piece.min()) < -1e-6 or float(piece.max()) > 1.000001:
                    raise ValueError('Invalid normalized video range')
        if not torch.isfinite(labels).all() or float(labels.min()) < -1e-6 or float(labels.max()) > 1.000001:
            raise ValueError('Invalid normalized BVP')
        results[split]={'feature_shape':list(features.shape),'label_shape':list(labels.shape),
                        'dtype':str(features.dtype),'model_clip_shape':[4,128,64,64],
                        'clips':int(features.shape[0]*(features.shape[1]//128)),
                        'feature_min':minimum,'feature_max':maximum,'all_values_finite':True}
    for recording in manifest['recordings']:
        audit_path=output_dir/'frame_indices'/recording['frame_audit']
        if sha256(audit_path)!=recording['frame_audit_sha256']: raise ValueError('Frame-index audit checksum mismatch')
        frames=json.loads(audit_path.read_text(encoding='utf-8'))
        if len(frames)!=recording['frames']: raise ValueError('Missing selected-frame audit rows')
        if [x['selected_frame'] for x in frames]!=list(range(len(frames))): raise ValueError('Selected frame order mismatch')
        if any(x['bvp_index']!=x['rgb_index'] for x in frames): raise ValueError('BVP/RGB index mismatch')
        if any(a['rgb_index']>=b['rgb_index'] for a,b in zip(frames,frames[1:])): raise ValueError('RGB selection is not strictly chronological')
    train={r['subject'] for r in manifest['row_map'] if r['split']=='train'}
    test={r['subject'] for r in manifest['row_map'] if r['split']=='test'}
    if train & test: raise ValueError('Subject leakage in exported train/test data')
    for split in ('train','test'):
        rows=[r for r in manifest['row_map'] if r['split']==split]
        if [r['row'] for r in rows]!=list(range(results[split]['feature_shape'][0])):
            raise ValueError('Non-contiguous row mapping')
    receipt={'status':'passed','checks':'output SHA256s, all tensor shapes/dtypes/values, clip boundaries and subject-disjoint rows',
             'outputs':results,'train_subjects':sorted(train),'test_subjects':sorted(test),
             'manifest_sha256':sha256(output_dir/'manifest.json'),'verifier_sha256':sha256(__file__)}
    (output_dir/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--setup',action='store_true',help='Install pinned dependencies and face model in an isolated Python3.12 environment')
    p.add_argument('--env-dir',type=Path,default=Path(__file__).resolve().parent/'.ibvp-venv',help='Managed environment location; default beside this file')
    p.add_argument('--use-current-env',action='store_true',help='Use your existing environment instead of automatic managed-environment launch')
    p.add_argument('--input-dir',type=Path,help='Raw iBVP_Dataset root (ZIPs or extracted streams)')
    p.add_argument('--output-dir',required=True,type=Path,help='New/empty output directory')
    p.add_argument('--sessions',nargs='+',help='Optional session subset, e.g. p32_d p22_a, for a real-data smoke run')
    p.add_argument('--train-subjects',nargs='+',default=PAPER_TRAIN)
    p.add_argument('--test-subjects',nargs='+',default=PAPER_TEST)
    p.add_argument('--frames',type=int,default=5376,help='Selected frames per raw recording (paper:5376)')
    p.add_argument('--row-frames',type=int,default=1792,help='Frames per saved row;5376/1792 gives3rows without dropping frames')
    p.add_argument('--sampling',choices=['legacy-random','uniform'],default='legacy-random')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--alignment',choices=['index','timestamp'],default='index')
    p.add_argument('--max-pair-offset-ms',type=float,default=50)
    p.add_argument('--normalization',choices=['split','recording'],default='split')
    p.add_argument('--face-backend',choices=['auto','legacy','tasks'],default='auto')
    p.add_argument('--face-model',type=Path,help='Optional Tasks FaceDetector .tflite; setup downloads a verified default')
    p.add_argument('--face-confidence',type=float,default=.5)
    p.add_argument('--missing-face',choices=['zero','error'],default='zero')
    p.add_argument('--max-missing-face-fraction',type=float,default=1.,help='Default keeps historical zero placeholders; lower to enforce a quality cutoff')
    p.add_argument('--thermal-width',type=int,default=640)
    p.add_argument('--thermal-height',type=int,default=512)
    p.add_argument('--bvp-column',default='BVP')
    p.add_argument('--analysis-rate',type=float,default=28.,help='Metadata only, matching current AMPNet config; does not resample28Hz')
    p.add_argument('--threads',type=int,default=4,help='CPU threads for torch and OpenCV')
    p.add_argument('--inspect',action='store_true',help='Report roster, availability and disk estimate without decoding frames')
    p.add_argument('--verify-only',action='store_true',help='Verify previously completed outputs, without rereading raw recordings')
    return p


def run(args):
    import cv2
    import torch
    if args.threads<=0: raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads);cv2.setNumThreads(args.threads)
    if args.verify_only:
        print(json.dumps(verify_output(args.output_dir),indent=2));return
    if args.input_dir is None or not args.input_dir.is_dir():
        raise ValueError('--input-dir must be an existing iBVP_Dataset directory')
    if args.frames<=0 or args.row_frames<=0 or args.frames%args.row_frames or args.row_frames%128:
        raise ValueError('--frames must divide into complete --row-frames rows; each row must be a multiple of128')
    if not (0<args.face_confidence<=1) or not (0<=args.max_missing_face_fraction<=1):
        raise ValueError('Confidence must be in(0,1]; missing-face fraction in[0,1]')
    if args.thermal_width<=0 or args.thermal_height<=0 or args.analysis_rate<=0 or args.max_pair_offset_ms<0:
        raise ValueError('Invalid dimensions, analysis rate or timestamp tolerance')
    raw_root=args.input_dir.resolve();out=args.output_dir.resolve()
    if out==raw_root or raw_root in out.parents:
        raise ValueError('Keep --output-dir outside the raw dataset tree')
    all_sessions=discover_sessions(raw_root)
    train_subjects=[s.lower() for s in args.train_subjects]
    test_subjects=[s.lower() for s in args.test_subjects]
    sessions=resolve_split(all_sessions,train_subjects,test_subjects)
    if args.sessions:
        wanted=set(args.sessions)
        missing=wanted-{s['session'] for s in sessions}
        if missing: raise ValueError(f'Sessions not in the selected subject split: {sorted(missing)}')
        sessions=[s for s in sessions if s['session'] in wanted]
    counts={split:sum(s['split']==split for s in sessions) for split in ('train','test')}
    if min(counts.values())==0: raise ValueError('Select at least one training and one test recording to export all four files')
    for s in sessions:
        stream_path(s,'_rgb');stream_path(s,'_t')
        values=read_bvp(Path(s['bvp']),args.bvp_column)
        if len(values)<args.frames: raise ValueError(f"Short BVP in {s['session']}")
    feature_bytes=len(sessions)*args.frames*64*64*4*4
    label_bytes=len(sessions)*args.frames*4
    peak_estimate=int((feature_bytes+label_bytes)*2.05+100*1024**2)
    ancestor=out
    while not ancestor.exists(): ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free
    plan={'raw_root':str(raw_root),'output_dir':str(out),'recordings':len(sessions),'recordings_per_split':counts,
          'selected_frames_per_recording':args.frames,'rows_per_recording':args.frames//args.row_frames,
          'clips_per_recording':args.frames//128,'output_feature_gib':feature_bytes/1024**3,
          'conservative_peak_disk_gib':peak_estimate/1024**3,'free_disk_gib':free/1024**3,
          'sessions':[{k:s[k] for k in ('subject','session','split','confidential_no_media_use')} for s in sessions]}
    print(json.dumps(plan,indent=2),flush=True)
    if args.inspect: return
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory is not empty; choose a new path (raw and previous outputs are never overwritten)')
    if peak_estimate>free:
        raise ValueError(f'Insufficient free disk: allow approximately{peak_estimate/1024**3:.1f}GiB; choose an output drive with more space or a --sessions subset')
    locator=FaceLocator(args.face_backend,args.face_model,args.face_confidence)
    out.mkdir(parents=True,exist_ok=True)
    work=out/'.work';work.mkdir()
    audit_dir=out/'frame_indices';audit_dir.mkdir()
    manifest={'converter_version':VERSION,'converter_sha256':sha256(__file__),
              'status':'processing','configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'actual_face_backend':locator.backend,'mediapipe_version':locator.version,
              'face_model_sha256':None if args.face_model is None or locator.backend=='legacy' else sha256(args.face_model),
              'dependencies':{'numpy':np.__version__,'opencv':cv2.__version__,'torch':torch.__version__},
              'plan':plan,'outputs':{},'row_map':[],'recordings':[],
              'method_notes':[
                  'Paper dimensions retained;1792-row packaging does not downsample5376selectedframes.',
                  'Paper table has18training and4test IDs; narrative count19training is inconsistent.',
                  'Sampling seed, normalization scope and detector revision are explicit implementation choices, not recovered original settings.',
                  'Index alignment follows archived code; timestamp alternative pairs thermal toRGBtimestamps and assumes BVProws correspond toRGBframes.',
                  'analysis_rate28Hz is metadata for the existing evaluator, not a claim of28Hz uniform sampling.',
                  'Missing faces use zero placeholders unless configured to fail; counts are recorded.',
                  'Confidential_No-Media-Use records remain local; this converter exports no face preview images.']}
    start=time.perf_counter()
    try:
        for split in ('train','test'):
            subset=[s for s in sessions if s['split']==split]
            rows=len(subset)*(args.frames//args.row_frames)
            fshape=(rows,args.row_frames,64,64,4);lshape=(rows,args.row_frames)
            fpath=work/(split+'_features.bin');lpath=work/(split+'_labels.bin')
            features=np.memmap(fpath,mode='w+',dtype=np.float32,shape=fshape)
            labels=np.memmap(lpath,mode='w+',dtype=np.float32,shape=lshape)
            flat_f=features.reshape(len(subset),args.frames,64,64,4)
            flat_l=labels.reshape(len(subset),args.frames)
            for idx,session in enumerate(subset):
                print(f"Processing {split}: {session['session']} ({idx+1}/{len(subset)})",flush=True)
                detail=process_session(session,args,locator,flat_f[idx],flat_l[idx],audit_dir)
                manifest['recordings'].append(detail)
                for part in range(args.frames//args.row_frames):
                    manifest['row_map'].append({'split':split,'row':idx*(args.frames//args.row_frames)+part,
                        'subject':session['subject'],'session':session['session'],
                        'selected_frame_start':part*args.row_frames,'selected_frame_stop':(part+1)*args.row_frames})
            if args.normalization=='split':
                stats=array_stats(features,labels)
                for row in range(rows): normalize_inplace(features[row],labels[row],stats)
            else: stats={'scope':'recording','details':'see each recording raw_normalization_stats'}
            features.flush();labels.flush()
            feature_name=f'ibvp_{split}_features.pth';label_name=f'ibvp_{split}_labels.pth'
            print(f'Saving {split} tensors...',flush=True)
            torch.save(torch.from_numpy(features),out/feature_name)
            torch.save(torch.from_numpy(labels),out/label_name)
            manifest['outputs'][split]={'features':feature_name,'labels':label_name,'feature_shape':list(fshape),
                'label_shape':list(lshape),'normalization_stats':stats,
                'sha256':{feature_name:sha256(out/feature_name),label_name:sha256(out/label_name)}}
            del flat_f,flat_l
            features._mmap.close();labels._mmap.close()
            del features,labels
            # Only remove temporary files this invocation created inside its new output directory.
            for path in (fpath,lpath):
                if path.resolve().parent != work.resolve(): raise RuntimeError('Unexpected staging path')
                path.unlink()
        work.rmdir()
        manifest['status']='complete';manifest['elapsed_seconds']=time.perf_counter()-start
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        receipt=verify_output(out)
        print(json.dumps({'status':'passed','output_dir':str(out),'recordings':len(sessions),'verification':receipt['outputs']},indent=2))
    except Exception as exc:
        manifest['status']='failed';manifest['error']=f'{type(exc).__name__}: {exc}'
        (out/'failure.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        raise
    finally:
        locator.close()


def main():
    try:
        args=parser().parse_args()
        check_setup_python()
        if np is None:
            raise RuntimeError('Dependencies are missing. Run: python prepare_ibvp.py --setup')
        if args.face_model is None and not args.use_current_env:
            candidate=args.env_dir.expanduser().resolve()/'models'/MODEL_NAME
            if candidate.is_file():
                if hashlib.sha256(candidate.read_bytes()).hexdigest()!=MODEL_SHA256:
                    raise ValueError('Managed face-model checksum mismatch. Repair the setup before processing.')
                args.face_model=candidate
        run(args)
    except (ValueError,OSError,RuntimeError,ImportError,zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
