#!/usr/bin/env python3
"""Run LAMP and optional LAMP-F on images with frozen CLIP features.

Prepare a support bank with prepare_data.py, then provide its manifest,
the dataset image root, and the matching OpenAI CLIP checkpoint.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, __version__ as pillow_version
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from clip_model import build_model
from kernels import global_kernel, prior, residual, set_kernels
from portable_math import global_gram, region_gram, solve, numerical_residual, global_response, configure, center_support

HERE = Path(__file__).resolve().parent
EQUAL = (1/3, 1/3, 1/3)
RHOS = (.025, .05, .1, .25, .5, .75, 1.)
RATIOS = (1., .5, 2.)
WEIGHTS = (EQUAL,) + tuple((a/4, b/4, (4-a-b)/4) for a in range(5) for b in range(5-a))
NORMALIZE = T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))
PREPROCESS = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(224), T.ToTensor(), NORMALIZE])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(cpu(v) for v in value)
    return value


def device_copy(value, device):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {k: device_copy(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(device_copy(v, device) for v in value)
    return value


class Images(Dataset):
    def __init__(self, root, bank, phase):
        self.root, self.bank, self.phase = root, bank, phase
        self.rows = (list(zip(bank['support_ids'], bank['support_labels'])) if phase == 'support'
                     else bank[phase])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        name, label = self.rows[i][:2]
        path = self.root / name
        if sha(path) != self.bank['image_sha256'][name]:
            raise RuntimeError('Image checksum differs: ' + name)
        with Image.open(path) as source: image = source.convert('RGB')
        flip = False
        if self.phase == 'support':
            width, height, top, left, crop_h, crop_w, flip = self.bank['support_geometry'][i]
            if image.size != (width, height): raise RuntimeError('Image size differs: ' + name)
            image = TF.resized_crop(image, top, left, crop_h, crop_w, [224, 224],
                                    T.InterpolationMode.BICUBIC, antialias=True)
            if flip: image = TF.hflip(image)
            pixels = NORMALIZE(TF.to_tensor(image))
        else:
            pixels = PREPROCESS(image)
        return pixels, label, bool(flip)


class Encoder:
    def __init__(self, model, backbone):
        self.backbone = backbone
        self.model, self.tokens, self.handles = model, {}, []
        if backbone == "RN50":
            for depth, name in enumerate(("layer2", "layer3", "layer4")):
                def capture(module, inputs, output, depth=depth): self.tokens[depth] = output
                self.handles.append(getattr(model.visual, name).register_forward_hook(capture))
            return
        for depth in (4, 8, 12):
            def capture(module, inputs, output, depth=depth):
                self.tokens[depth] = output[1:].permute(1, 0, 2)
            self.handles.append(model.visual.transformer.resblocks[depth-1].register_forward_hook(capture))

    @torch.inference_mode()
    def __call__(self, pixels, flips):
        self.tokens.clear()
        raw = self.model.encode_image(pixels).float()
        outputs = []
        if self.backbone == "RN50":
            for depth in range(3):
                x = self.tokens[depth]
                pooled = F.adaptive_avg_pool2d(x.clamp_min(0).pow(3), (2, 2)).clamp_min(0).pow(1/3)
                outputs.append(F.normalize(pooled.flatten(2).transpose(1, 2), dim=-1))
            self.tokens.clear()
            return {"features": F.normalize(raw, dim=-1), "native": outputs}
        for depth in (4, 8, 12):
            token = self.tokens[depth]
            native = F.normalize(F.layer_norm(token, (768,)).transpose(1, 2).reshape(len(token), 768, 14, 14), dim=1)
            aligned = torch.where(flips[:, None, None, None], native.flip(-1), native)
            outputs.append(F.normalize(F.adaptive_avg_pool2d(aligned, (2, 2)).flatten(2).transpose(1, 2), dim=-1))
        result = {'features': raw / raw.norm(dim=-1, keepdim=True), 'native': outputs}
        self.tokens.clear()
        return result

    def close(self):
        for handle in self.handles: handle.remove()


@torch.inference_mode()
def encode(args, bank, phases):
    checkpoint = torch.jit.load(str(args.checkpoint), map_location='cpu').eval()
    model = build_model(checkpoint.state_dict()).to(args.device).float().eval().requires_grad_(False)
    del checkpoint
    encoder = Encoder(model, args.backbone)
    # Prepared prompt tokens keep tokenization consistent between stages.
    columns = []
    for tokens in bank['prompt_token_ids']:
        embedding = model.encode_text(torch.tensor(tokens, device=args.device)).float()
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.mean(dim=0)
        embedding /= embedding.norm()
        columns.append(embedding)
    text = torch.stack(columns, dim=1).cpu().contiguous()
    if (args.output / 'text.pt').is_file():
        if not torch.equal(text, torch.load(args.output / 'text.pt', weights_only=True)):
            raise RuntimeError('Re-encoded prompts changed between stages')
    else:
        torch.save(text, args.output / 'text.pt')
    for phase in phases:
        dataset = Images(args.images, bank, phase)
        if phase == 'support':
            n = len(bank['classes']) * args.shot
            batches = [list(range(start, min(start + args.batch_size, view+n)))
                       for view in range(0, len(dataset), n)
                       for start in range(view, view+n, args.batch_size)]
            loader = DataLoader(dataset, batch_sampler=batches, num_workers=args.workers, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        values, labels, natives = [], [], [[], [], []]
        began = time.perf_counter()
        for pixels, y, flips in loader:
            out = encoder(pixels.to(args.device), flips.to(args.device))
            values.append(out['features'].cpu()); labels.append(y)
            for target, native in zip(natives, out['native']): target.append(native.cpu())
        encoded = {'features': torch.cat(values), 'native': [torch.cat(x) for x in natives],
                   'labels': torch.cat(labels), 'bank_sha256': args.bank_sha256}
        for value in [encoded['features'], *encoded['native']]:
            if not torch.isfinite(value).all() or (value.norm(dim=-1)-1).abs().max() >= 2e-5:
                raise RuntimeError('Nonfinite or nonunit feature')
        torch.save(encoded, args.output / (phase + '.pt'))
        print(f'Encoded {phase}: {len(dataset)} rows, {time.perf_counter()-began:.2f}s', flush=True)
    encoder.close()


def old_dense_center_support(gram):
    means = gram.sum(1, dtype=torch.float64) / len(gram)
    mean = means.mean()
    var = gram.diagonal().double() - 2 * means + mean
    if float(var.min()) < -2e-6: raise RuntimeError('Negative support variance')
    scale = var.clamp_min(1e-10).sqrt()
    result = ((gram.double() - means[:, None] - means[None] + mean) /
              (scale[:, None] * scale[None])).float()
    return result, dict(means=means, mean=mean, scale=scale)


def center_query(cross, state):
    x = cross.double(); qm = x.mean(1)
    var = 1 - 2 * qm + state['mean']
    if float(var.min()) < -2e-6: raise RuntimeError('Negative query variance')
    return ((x - qm[:, None] - state['means'][None] + state['mean']) /
            (var.clamp_min(1e-10).sqrt()[:, None] * state['scale'][None])).float()


def native_delta(q, s, branch):
    outputs = []
    for begin in range(0, len(q), 512):
        cross = set_kernels(q[begin:begin+512], s, [branch['beta']], 128, 128)[0]
        outputs.append(center_query(cross, branch['state']) @ branch['alpha'])
    return torch.cat(outputs)


def correct(scores, labels):
    return int((scores.topk(1, 1, True, True).indices.flatten() == labels).sum())


def mix(p, global_delta, terms, row):
    if row['rho'] == 0: return p + global_delta
    w = row['weights']
    term = (sum(terms) / 3 if tuple(w) == EQUAL else
            float(w[0]) * terms[0] + float(w[1]) * terms[1] + float(w[2]) * terms[2])
    return p + (1 - row['rho']) * global_delta + row['rho'] * term


def choice(row):
    return (-row['validation_correct'], row['rho'], abs(row['ratio']-1), row['ratio'],
            sum((x-1/3)**2 for x in row['weights']), bool(row['rho']), tuple(row['weights']))


def load_feature(args, name):
    value = torch.load(args.output / (name + '.pt'), map_location='cpu', weights_only=True)
    if isinstance(value, dict) and value['bank_sha256'] != args.bank_sha256:
        raise RuntimeError('Feature cache belongs to a different bank')
    return device_copy(value, args.device)


@torch.inference_mode()
def select(args):
    s, q, text = [load_feature(args, x) for x in ('support', 'validation', 'text')]
    target, p = residual(s['features'], s['labels'], text), prior(q['features'], text)
    best, rows = None, []
    began = time.perf_counter()
    grid = json.loads((HERE / 'configs' / 'global_grid.json').read_text())['datasets'][args.dataset]
    expected_global = grid['candidate_count']
    for beta in torch.linspace(*grid['beta'], dtype=torch.float32).tolist():
        gram = global_gram(s['features'], beta)
        cross = global_kernel(q['features'], s['features'], beta) if len(s['features']) <= args.dense_limit else None
        for lam in torch.linspace(*grid['lambda_'], dtype=torch.float32).tolist():
            alpha = solve(gram, target, lam)
            delta = cross @ alpha if cross is not None else global_response(q['features'], s['features'], alpha, beta)
            row = dict(beta=beta, **{'lambda': lam}, validation_correct=correct(p + delta, q['labels']))
            rows.append(row)
            # Ascending beta/lambda traversal selects the first maximum.
            if best is None or row['validation_correct'] > best['row']['validation_correct']:
                best = dict(row=row, alpha=alpha.clone(), delta=delta.clone())
        print(f'Global candidates {len(rows)}/{expected_global}; best={best["row"]["validation_correct"]}', flush=True)
    write(args.output / 'global_candidates.json', rows)
    beta, lam = best['row']['beta'], best['row']['lambda']
    # Fixed selected kernels also supply the exact prototype response B = L S.
    response_global = solve(global_gram(s['features'], beta), s['features'], lam)
    zero = dict(rho=0., ratio=1., weights=list(EQUAL), validation_correct=best['row']['validation_correct'])
    chosen, regional_rows, branches, diagnostics = zero, [zero], {}, {}
    for ratio in RATIOS:
        terms = []
        for index, (xs, xq) in enumerate(zip(s['native'], q['native'])):
            gram = region_gram(xs, beta*ratio)
            gram, state = center_support(gram)
            alpha = solve(gram, target, lam)
            check = numerical_residual(gram, target, alpha, lam)
            if check['backward_error'] >= 1e-6: raise RuntimeError('Unstable regional solve')
            branch = dict(alpha=alpha, state=state, beta=beta*ratio, response=solve(gram, s['features'], lam))
            branches[(ratio, index)] = branch
            diagnostics[str((ratio, index))] = check
            terms.append(native_delta(xq, xs, branch))
        for rho in RHOS:
            for w in WEIGHTS:
                row = dict(rho=rho, ratio=ratio, weights=list(w))
                row['validation_correct'] = correct(mix(p, best['delta'], terms, row), q['labels'])
                regional_rows.append(row)
                if choice(row) < choice(chosen): chosen = row
    if len(rows) != expected_global or len(regional_rows) != 337: raise RuntimeError('Incomplete search')
    write(args.output / 'regional_candidates.json', regional_rows)
    selection = dict(global_parameters=best['row'], regional_parameters=chosen,
                     global_candidates=expected_global, regional_candidates=337, validation_rows=len(q['labels']),
                     bank_sha256=args.bank_sha256, test_used_for_selection=args.selection_evaluation_overlap,
                     evaluation_scope='selection-overlapping test' if args.selection_evaluation_overlap else 'held-out test',
                     feature_sha256={name: sha(args.output / (name + '.pt')) for name in ('support', 'validation', 'text')},
                     fit_select_seconds=time.perf_counter()-began, diagnostics=diagnostics)
    selected_branches = [branches[(chosen['ratio'], i)] for i in range(3)] if chosen['rho'] else []
    torch.save(cpu(dict(alpha=best['alpha'], response_global=response_global, branches=selected_branches, selection=selection)), args.output / 'model.pt')
    selection['model_sha256'] = sha(args.output / 'model.pt')
    write(args.output / 'selection.json', selection)
    print('Selection sealed:', json.dumps(selection['regional_parameters']), flush=True)


@torch.inference_mode()
def evaluate(args):
    selection = json.loads((args.output / 'selection.json').read_text())
    if sha(args.output / 'model.pt') != selection['model_sha256']: raise RuntimeError('Selected model changed')
    for name, expected in selection['feature_sha256'].items():
        if sha(args.output / (name + '.pt')) != expected: raise RuntimeError('Fit input changed: ' + name)
    model = device_copy(torch.load(args.output / 'model.pt', weights_only=True, map_location='cpu'), args.device)
    s, q, text = [load_feature(args, x) for x in ('support', 'test', 'text')]
    row, beta = selection['regional_parameters'], selection['global_parameters']['beta']
    p = prior(q['features'], text)
    delta = global_response(q['features'], s['features'], model['alpha'], beta)
    terms = [native_delta(xq, xs, branch) for xq, xs, branch in zip(q['native'], s['native'], model['branches'])]
    scores = {'GLOBAL': p + delta, 'LAMP': mix(p, delta, terms, row)}
    if args.method == 'both':
        adapted = load_adapted(args)
        response = response_scores(q, s, model, selection)
        delta = adapted['delta'].to(args.device)
        scores['LAMP_F'] = scores['LAMP'] + q['features'] @ delta.T - response @ delta.T
    test = {name: dict(correct=correct(value, q['labels']), count=len(q['labels']),
                       top1=100*correct(value, q['labels'])/len(q['labels'])) for name, value in scores.items()}
    np.savez_compressed(args.output / 'test_predictions.npz', labels=q['labels'].cpu().numpy(),
                        **{name: value.topk(1, 1, True, True).indices.flatten().cpu().numpy() for name, value in scores.items()})
    report = dict(dataset=args.dataset, backbone=args.backbone, shot=args.shot, seed=args.seed,
                  full_validation_rows=selection['validation_rows'], full_test_rows=len(q['labels']),
                  selection=selection, test_accuracy=test,
                  selection_sha256=sha(args.output / 'selection.json'),
                  prediction_sha256=sha(args.output / 'test_predictions.npz'))
    write(args.output / 'result.json', report)
    print(json.dumps(report, indent=2), flush=True)


def response_scores(q, s, model, selection):
    beta = selection['global_parameters']['beta']
    g_response = global_response(q['features'], s['features'], model['response_global'], beta)
    terms = [native_delta(xq, xs, dict(branch, alpha=branch['response']))
             for xq, xs, branch in zip(q['native'], s['native'], model['branches'])]
    return mix(torch.zeros_like(g_response), g_response, terms, selection['regional_parameters'])


def free_scores(q, s, text, model, selection):
    p = prior(q['features'], text)
    gd = global_response(q['features'], s['features'], model['alpha'], selection['global_parameters']['beta'])
    terms = [native_delta(xq, xs, branch) for xq, xs, branch in zip(q['native'], s['native'], model['branches'])]
    return mix(p, gd, terms, selection['regional_parameters'])


def batch_plan(count, seed):
    if count < 64: raise ValueError('Prototype training requires at least 64 support views')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loader = DataLoader(torch.utils.data.TensorDataset(torch.arange(count)), batch_size=64,
                            shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        iterator, result = iter(loader), []
        for _ in range(200):
            try: item, = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                item, = next(iterator)
            result.append(item)
    return torch.stack(result)


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """Fifty warmup steps followed by a 200-step cosine horizon."""
    def __init__(self, optimizer):
        self.successor = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200.)
        super().__init__(optimizer, last_epoch=-1)
    def get_lr(self):
        if self.last_epoch >= 50: return self.successor.get_last_lr()
        if self.last_epoch == 0: return [1e-5]
        return [lr * self.last_epoch / 50 for lr in self.base_lrs]
    def step(self, epoch=None):
        if self.last_epoch >= 50:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else: super().step(epoch)


def adapt(args, bank):
    selection = json.loads((args.output / 'selection.json').read_text())
    if sha(args.output / 'model.pt') != selection['model_sha256']: raise RuntimeError('Kernel model changed')
    s, q, text = [load_feature(args, name) for name in ('support','validation','text')]
    model = device_copy(torch.load(args.output / 'model.pt', weights_only=True), args.device)
    with torch.no_grad():
        free = free_scores(q, s, text, model, selection)
        response = response_scores(q, s, model, selection)
        text = text.T.contiguous()
        direction = F.normalize(text, dim=-1)
        logits = s['features'].float() @ text.float().T
        target = F.one_hot(s['labels'], num_classes=len(text)).to(logits.dtype)
        anchor_weights = (torch.diag(logits.softmax(-1).T @ target) / target.sum(0)).detach()
        # Image identities explicitly determine the ten-view arithmetic mean.
        groups = {}
        for i, key in enumerate(bank['support_ids']): groups.setdefault(key, []).append(i)
        if not all(len(x) == 10 for x in groups.values()): raise RuntimeError('Not ten views per image')
        index = torch.tensor(list(groups.values()), device=args.device)
        means = s['features'][index].mean(1)
        mean_rows = torch.empty_like(s['features']); mean_rows[index] = means[:,None,:]
    plan = batch_plan(len(s['labels']), args.seed)
    plan_sha = hashlib.sha256(plan.numpy().tobytes()).hexdigest()
    plan = plan.to(args.device)
    best, curves, candidates, rates = None, [], [], []
    for learning_rate in (3e-5, 1e-4, 3e-4):
        prototype = torch.nn.Parameter(text.detach().clone())
        optimizer = torch.optim.AdamW([dict(params=[prototype], name='prototypes', weight_decay=.01)], lr=learning_rate)
        scheduler = WarmupCosine(optimizer)
        for step in range(201):
            row = dict(learning_rate=learning_rate, step=step)
            if step:
                optimizer.zero_grad(set_to_none=True)
                chosen = plan[step-1]
                delta = F.normalize(prototype, dim=-1) - direction
                scores = mean_rows[chosen] @ (text + delta).T
                ce = F.cross_entropy(100 * scores, s['labels'][chosen])
                anchor = (anchor_weights * (prototype-text).square().sum(-1)).mean()
                loss = ce + anchor
                loss.backward()
                if not torch.isfinite(prototype.grad).all(): raise RuntimeError('Nonfinite gradient')
                actual_lr = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                row.update(train_loss=float(loss.detach()), train_ce=float(ce.detach()),
                           train_constraint=float(anchor.detach()), prototype_lr=actual_lr)
                rates.append(dict(learning_rate=learning_rate,step=step,prototype_lr=actual_lr))
            if step % 20 == 0:
                with torch.no_grad():
                    delta = F.normalize(prototype, dim=-1) - direction
                    scores = free + q['features'] @ delta.T - response @ delta.T
                    if step == 0 and not torch.equal(scores,free): raise RuntimeError('Step-zero equivalence fails')
                    row['validation_correct'] = correct(scores, q['labels'])
                    candidate = dict(learning_rate=learning_rate,step=step,validation_correct=row['validation_correct'])
                    candidates.append(candidate)
                    key = lambda x: (-x['validation_correct'],x['step'],x['learning_rate'])
                    if best is None or key(candidate) < key(best['selected']):
                        best = dict(selected=candidate, prototype=prototype.detach().cpu().clone(),
                                    delta=delta.cpu().clone())
            curves.append(row)
        print('Prototype trial',learning_rate,'selected',best['selected'],flush=True)
    if len(candidates) != 33: raise RuntimeError('Incomplete prototype search')
    write(args.output/'prototype_candidates.json',candidates)
    write(args.output/'prototype_curves.json',curves)
    write(args.output/'learning_rate_trace.json',rates)
    torch.save(best,args.output/'prototype.pt')
    seal = dict(selected=best['selected'],candidates=33,minibatches_sha256=plan_sha,
                support_only_updates=True,step_zero_exact=True,kernel_selection_sha256=sha(args.output/'selection.json'),
                prototype_sha256=sha(args.output/'prototype.pt'))
    write(args.output/'prototype_selection.json',seal)


def load_adapted(args):
    seal = json.loads((args.output/'prototype_selection.json').read_text())
    if sha(args.output/'selection.json') != seal['kernel_selection_sha256']: raise RuntimeError('Kernel selection changed')
    if sha(args.output/'prototype.pt') != seal['prototype_sha256']: raise RuntimeError('Selected prototype changed')
    return torch.load(args.output/'prototype.pt',map_location='cpu',weights_only=True)


def load_bank(args):
    path = args.bank_file
    with gzip.open(path, 'rt') as f: data = json.load(f)
    preparation = data.get('preparation', {})
    for key in ('dataset', 'shot', 'seed'):
        if key in preparation and preparation[key] != getattr(args, key):
            raise RuntimeError('Support bank does not match --' + key)
    support = data['support_banks'][f'{args.shot}_{args.seed}']
    bank = {k: data[k] for k in ('classes','validation','test','image_sha256','prompt_token_ids')}
    bank.update(support_ids=support['ids'] * 10, support_labels=support['labels'] * 10,
                support_geometry=support['geometry'])
    overlap = bool({row[0] for row in bank['validation']} & {row[0] for row in bank['test']})
    if 'selection_evaluation_overlap' in data and bool(data['selection_evaluation_overlap']) != overlap:
        raise RuntimeError('Manifest validation/test overlap flag disagrees with image identities')
    args.selection_evaluation_overlap = overlap
    args.bank_sha256 = hashlib.sha256((sha(path)+f'/{args.shot}/{args.seed}/{args.backbone}').encode()).hexdigest()
    # Verify only the selected support images and the two complete query splits.
    needed = set(bank['support_ids']) | {r[0] for p in ('validation','test') for r in bank[p]}
    bank['image_sha256'] = {name: data['image_sha256'][name] for name in needed}
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', choices=['ViT-B/16','RN50'], required=True)
    parser.add_argument('--shot', type=int, choices=[1,2,4,8,16], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Dataset image root used by prepare_data.py')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--method', choices=['lamp','both'], default='lamp',
                        help='lamp: training-free prediction; both: also train LAMP-F prototypes')
    parser.add_argument('--stage', choices=['verify','encode','select','adapt','evaluate','all'], default='all')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dense-limit', type=int, default=12000)
    parser.add_argument('--lu-tile', type=int, default=2048)
    parser.add_argument('--bank-file', type=Path, required=True,
                        help='Gzip JSON support-bank manifest generated by prepare_data.py')
    args = parser.parse_args()
    bank = load_bank(args)
    args.output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((HERE / 'configs' / 'checkpoints.json').read_text())['checkpoint_sha256'][args.backbone]
    if sha(args.checkpoint) != expected: raise RuntimeError('Wrong checkpoint')
    for identity, expected in bank['image_sha256'].items():
        if sha(args.images / identity) != expected: raise RuntimeError('Image missing or changed: ' + identity)
    print(f"Verified checkpoint and {len(bank['image_sha256'])} unique raw images", flush=True)
    if args.stage == 'verify': return
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('LAMP encoding and kernel fitting require a CUDA device')
    torch.set_num_threads(4); torch.cuda.set_device(args.device)
    configure(args.output,args.dense_limit,args.lu_tile)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.backbone != 'RN50'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.backbone == 'RN50'
    runtime = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                   numpy=np.__version__, pillow=pillow_version, cuda_build=torch.version.cuda,
                   device=torch.cuda.get_device_name(args.device), batch_size=args.batch_size, workers=args.workers,
                   source_sha256={x.name: sha(x) for x in HERE.glob('*.py')}, bank_sha256=args.bank_sha256,
                   config_sha256={name: sha(HERE / 'configs' / name) for name in ('global_grid.json', 'checkpoints.json')},
                   invocation={k:str(v) for k,v in vars(args).items()})
    previous = args.output / 'runtime.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        for field in ('source_sha256','config_sha256','bank_sha256','batch_size'):
            if old[field] != runtime[field]: raise RuntimeError('Resumed execution inputs changed: ' + field)
    else: write(previous, runtime)
    if args.stage in ('encode', 'all'): encode(args, bank, ['support', 'validation'])
    if args.stage in ('select', 'all'): select(args)
    if args.method == 'both' and args.stage in ('adapt', 'all'): adapt(args, bank)
    if args.stage in ('evaluate', 'all'):
        if not (args.output / 'selection.json').is_file(): raise RuntimeError('Seal validation selection first')
        if args.method == 'both': load_adapted(args)
        encode(args, bank, ['test'])
        evaluate(args)


if __name__ == '__main__': main()
