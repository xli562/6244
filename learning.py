"""Collect expert RGBD demonstrations, train action-chunk imitation, and evaluate."""

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from expert import CONTROL_DT, Expert, apply_control, proprioception
from fetch_assets import SOURCES
from progress import add_progress_arguments, progress_bar
from simulation import KitchenSimulation, ORIENTATIONS, SceneConfig


def collect(args) -> None:
    """Write successful demonstrations and a manifest containing every attempted trial."""
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Choose a new dataset directory: {args.output}')
    manifest = dict(version=1, randomization=args.randomization, control_dt=CONTROL_DT,
                    observation_keys=['rgb', 'depth', 'proprioception'],
                    mujoco_version=mujoco.__version__, asset_sources=SOURCES,
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ['simulation.py', 'expert.py']}, episodes=[])
    trials = progress_bar(range(args.episodes), desc='Collect', unit='trial',
                          disabled=getattr(args, 'no_progress', False))
    for index in trials:
        seed = args.seed_start + index if args.randomization != 'none' else args.seed_start
        config = SceneConfig.randomized(seed) if args.randomization == 'scene' else SceneConfig()
        orientation = ORIENTATIONS[index % 3] if args.randomization != 'none' else 'upright'
        sim = KitchenSimulation(config, seed, orientation)
        frames = dict(rgb=[], depth=[], proprioception=[], actions=[], stage=[])
        last_stage = None

        def record(simulation, action, stage):
            """Record only camera observations, robot state, and expert actuator labels."""
            nonlocal last_stage
            if not trials.disable and stage != last_stage:
                trials.set_postfix_str(f'{seed} {stage.replace("_", " ")}')
                last_stage = stage
            observation = simulation.observe(args.width, args.height)
            frames['rgb'].append(observation['rgb'])
            frames['depth'].append(observation['depth'].astype(np.float16))
            frames['proprioception'].append(proprioception(simulation))
            frames['actions'].append(action.astype(np.float32))
            frames['stage'].append(stage)  # Diagnostic label; excluded from policy inputs.

        try:
            report = Expert(sim, record).run()
            filename = f'episode_{index:05d}.npz' if report['success'] else None
            entry = dict(file=filename, seed=seed, orientation=orientation,
                         config=asdict(config), report=report)
            if filename:
                calibration = sim.observe(args.width, args.height)
                arrays = {key: np.asarray(values) for key, values in frames.items()}
                arrays.update(intrinsics=calibration['intrinsics'],
                              camera_to_world=calibration['camera_to_world'])
                with (args.output / filename).open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
            manifest['episodes'].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            trials.set_postfix(result='success' if report['success'] else 'failed', refresh=False)
            if getattr(args, 'verbose', False):
                trials.write(f"{index + 1}/{args.episodes}: {orientation}, {config.handle}, "
                             f"success={report['success']}, failure={report['failure']}")
        finally:
            sim.close()
    successes = sum(entry['file'] is not None for entry in manifest['episodes'])
    print(f'Saved {successes}/{args.episodes} successful demonstrations to {args.output}')


class Policy(nn.Module):
    def __init__(self, image_shape, horizon, mean, std, history_length:int=10):
        """Build an RGBD encoder and action-chunk head with robot motion history."""
        super().__init__()
        self.horizon = horizon
        self.history_length = history_length
        self.register_buffer('proprio_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('proprio_std', torch.as_tensor(std, dtype=torch.float32))
        height, width = image_shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height),
                                       torch.linspace(-1, 1, width), indexing='ij')
        self.register_buffer('pixel_grid', torch.stack([grid_x, grid_y])[None])
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 7)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(48 * 5 * 7 + 26 * history_length, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, horizon * 8))

    def forward(self, image, proprio):
        """Map RGBD and robot state to normalized joint targets and gripper logits."""
        grid = self.pixel_grid.expand(image.shape[0], -1, -1, -1)
        visual = self.encoder(torch.cat([image, grid], dim=1))
        robot = ((proprio - self.proprio_mean) / self.proprio_std).flatten(1)
        return self.head(torch.cat([visual, robot], dim=1)).reshape(-1, self.horizon, 8)


class Demonstrations(Dataset):
    def __init__(self, paths, horizon, history_length:int=10, action_mean=None,
                 action_space:str='absolute', show_progress:bool=False,
                 description:str='Loading trajectories'):
        """Load trajectories and index causal robot histories with future action labels."""
        self.episodes = []
        self.indices = []
        self.horizon = horizon
        self.history_length = history_length
        self.action_mean = np.zeros(7) if action_mean is None else np.asarray(action_mean)
        self.action_space = action_space
        for path in progress_bar(paths, desc=description, unit='episode',
                                 disabled=not show_progress, leave=False):
            with np.load(path, allow_pickle=False) as source:
                # Explicit allowlist prevents diagnostic metadata becoming observations.
                episode = {key: source[key] for key in
                           ['rgb', 'depth', 'proprioception', 'actions']}
            episode_index = len(self.episodes)
            self.episodes.append(episode)
            self.indices.extend((episode_index, frame) for frame in range(len(episode['rgb'])))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """Return current RGBD, past/current robot states, and future actuator targets."""
        episode_index, frame = self.indices[index]
        episode = self.episodes[episode_index]
        image = prepare_image(episode['rgb'][frame], episode['depth'][frame])
        proprio = episode['proprioception'][frame]
        past = np.maximum(np.arange(frame - self.history_length + 1, frame + 1), 0)
        history = episode['proprioception'][past]
        future = np.minimum(np.arange(frame, frame + self.horizon), len(episode['actions']) - 1)
        targets = episode['actions'][future].copy()
        origin = proprio[:7] if self.action_space == 'delta' else self.action_mean
        targets[:, :7] = (targets[:, :7] - origin) / .25
        targets[:, 7] /= 255
        return image, history, targets

def prepare_image(rgb:np.ndarray, depth:np.ndarray) -> np.ndarray:
    """Return channel-first float32 RGBD, with metric depth clipped to three meters."""
    image = np.concatenate([rgb.astype(np.float32) / 255,
                            np.clip(depth.astype(np.float32), 0, 3)[..., None] / 3], axis=-1)
    return image.transpose(2, 0, 1).copy()


def split_episodes(episodes:list[dict], seed:int) -> tuple[list[dict], list[dict]]:
    """Split distinct initial configurations, keeping repeated resets in one partition."""
    groups = {}
    for entry in episodes:
        key = json.dumps([entry['seed'], entry['orientation'], entry['config']], sort_keys=True)
        groups.setdefault(key, []).append(entry)
    if len(groups) == 1:
        return episodes, []
    keys = list(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    count = max(1, round(len(keys) * .2))
    validation = [entry for index in order[:count] for entry in groups[keys[index]]]
    training = [entry for index in order[count:] for entry in groups[keys[index]]]
    return training, validation


def train(args) -> None:
    """Fit action-chunk behavior cloning with entire episodes held out for validation."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    episodes = [entry for entry in manifest['episodes'] if entry['file']]
    if not episodes:
        raise ValueError('At least one successful demonstration episode is required.')
    training_entries, validation_entries = split_episodes(episodes, args.seed)
    no_progress = getattr(args, 'no_progress', False)
    training = Demonstrations([args.data / entry['file'] for entry in training_entries],
                              args.horizon, action_space=args.action_space,
                              show_progress=not no_progress, description='Load training')
    validation = Demonstrations([args.data / entry['file'] for entry in validation_entries],
                                args.horizon, action_space=args.action_space,
                                show_progress=not no_progress, description='Load validation')
    action_mean = np.concatenate([episode['actions'][:, :7] for episode in training.episodes]).mean(0)
    training.action_mean = validation.action_mean = action_mean
    robot = np.concatenate([episode['proprioception'] for episode in training.episodes])
    mean, std = robot.mean(0), np.maximum(robot.std(0), .01)
    image_shape = list(training.episodes[0]['depth'].shape[1:])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = Policy(image_shape, args.horizon, mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    training_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size) if validation_entries else None
    history, best = [], float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Training on {device}: {len(training_entries)} training / {len(validation_entries)} validation episodes', flush=True)
    epochs = progress_bar(range(args.epochs), desc='Epochs', unit='epoch', disabled=no_progress)
    for epoch in epochs:
        losses = []
        loaders = [(training_loader, True)]
        if validation_loader is not None:
            loaders.append((validation_loader, False))
        for loader, update in loaders:
            model.train(update)
            total, count = 0., 0
            phase = 'Train' if update else 'Validate'
            with progress_bar(loader, desc=f'{phase} {epoch + 1}/{args.epochs}', unit='batch',
                              disabled=no_progress, leave=False, position=1) as batches:
                for image, proprio, target in batches:
                    image, proprio, target = image.to(device), proprio.to(device), target.to(device)
                    with torch.set_grad_enabled(update):
                        prediction = model(image, proprio)
                        loss = nn.functional.mse_loss(prediction[:, :, :7], target[:, :, :7])
                        loss += .1 * nn.functional.binary_cross_entropy_with_logits(
                            prediction[:, :, 7], target[:, :, 7])
                        if update:
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 5)
                            optimizer.step()
                    total += float(loss.detach()) * len(image)
                    count += len(image)
                    batches.set_postfix(loss=f'{total / count:.4f}', refresh=False)
            losses.append(total / count)
        validation_loss = losses[1] if validation_loader is not None else None
        history.append(dict(epoch=epoch + 1, train_loss=losses[0], validation_loss=validation_loss))
        selection_loss = validation_loss if validation_loss is not None else losses[0]
        if selection_loss < best:
            best = selection_loss
            checkpoint = dict(version=1, state=model.state_dict(), image_shape=image_shape,
                              horizon=args.horizon, mean=mean.tolist(), std=std.tolist(),
                              training_episodes=training_entries, validation_episodes=validation_entries,
                              randomization=manifest['randomization'], history=history,
                              input_keys=['rgb', 'depth', 'proprioception'], seed=args.seed,
                              history_length=model.history_length,
                              action_space=args.action_space, action_mean=action_mean.tolist(),
                              checkpoint_selection='validation_loss' if validation_loader else 'training_loss')
            temporary = args.output.with_suffix(args.output.suffix + '.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
        epochs.set_postfix(train=f'{losses[0]:.4f}',
                           val=f'{validation_loss:.4f}' if validation_loss is not None else 'n/a', refresh=False)
        if getattr(args, 'verbose', False):
            epochs.write(json.dumps(history[-1]))
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2))
    print(f'Saved best checkpoint to {args.output} ({device})')


def evaluate(args) -> None:
    """Run the learned policy on disjoint seeds with no expert fallback or scene-state input."""
    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = Policy(checkpoint['image_shape'], checkpoint['horizon'],
                   checkpoint['mean'], checkpoint['std'], checkpoint.get('history_length', 1)).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    used_seeds = {entry['seed'] for key in ['training_episodes', 'validation_episodes']
                  for entry in checkpoint[key]}
    requested = set(range(args.seed_start, args.seed_start + args.episodes))
    if used_seeds & requested and not args.allow_training_seeds:
        raise ValueError('Evaluation seeds overlap development data. Choose disjoint seeds.')
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(checkpoint=str(args.checkpoint), randomization=args.randomization,
                   seed_overlap=sorted(used_seeds & requested), episodes=reports, successes=0, trials=0,
                   control_dt=CONTROL_DT, execute=args.execute, seconds=args.seconds,
                   action_space=checkpoint.get('action_space', 'delta'))
    trials = progress_bar(range(args.episodes), desc='Eval', unit='trial',
                          disabled=getattr(args, 'no_progress', False))
    for index in trials:
        seed = args.seed_start + index
        config = SceneConfig.randomized(seed) if args.randomization == 'scene' else SceneConfig()
        orientation = ORIENTATIONS[index % 3]
        sim = KitchenSimulation(config, seed, orientation)
        robot_history = deque([proprioception(sim)] * model.history_length, maxlen=model.history_length)
        progress = dict(drawer_opened=False, cup_lifted=False, cup_placed=False, drawer_closed=False)
        failure = 'timeout'
        frames, next_frame, maximum_opening, next_progress = [], 0., 0., 0.
        try:
            while sim.data.time < args.seconds:
                if not trials.disable and sim.data.time >= next_progress:
                    trials.set_postfix_str(f'{seed} {sim.data.time:.1f}/{args.seconds:g} sim s | {summary["successes"]} ok')
                    next_progress = sim.data.time + 5
                height, width = checkpoint['image_shape']
                observation = sim.observe(width, height)
                if getattr(args, 'video', False) and sim.data.time >= next_frame:
                    from PIL import Image
                    frames.append(Image.fromarray(observation['rgb']))
                    next_frame += .5
                robot = proprioception(sim)
                image = prepare_image(observation['rgb'], observation['depth'])
                with torch.inference_mode():
                    prediction = model(torch.from_numpy(image[None]).to(device),
                                       torch.from_numpy(np.array(robot_history)[None]).to(device))[0].cpu().numpy()
                if not np.isfinite(prediction).all():
                    failure = 'non_finite_prediction'
                    break
                targets = prediction[:args.execute].copy()
                origin = checkpoint['action_mean'] if checkpoint.get('action_space') == 'absolute' else robot[:7]
                targets[:, :7] = targets[:, :7] * .25 + origin
                targets[:, 7] = (targets[:, 7] > 0) * 255
                for target in targets:
                    lower, upper = sim.model.actuator_ctrlrange.T
                    target[:7] = np.clip(target[:7], sim.data.ctrl[:7] - .15, sim.data.ctrl[:7] + .15)
                    apply_control(sim, np.clip(target, lower, upper))
                    robot_history.append(proprioception(sim))
                    # Privileged state is used only for evaluation, after the action.
                    status = sim.status()
                    maximum_opening = max(maximum_opening, status['drawer_opening'])
                    progress['drawer_opened'] |= status['drawer_opening'] > .28
                    minimum, _ = sim.cup_bounds()
                    progress['cup_lifted'] |= bool(progress['drawer_opened']
                                                  and minimum[2] > config.counter_height + .06)
                    progress['cup_placed'] |= progress['cup_lifted'] and status['cup_in_drawer']
                    progress['drawer_closed'] |= progress['cup_placed'] and status['success']
                    if sim.data.time >= args.seconds:
                        break
                if all(progress.values()) and status['success'] and sim.is_stable():
                    failure = None
                    break
                if sim.data.warning.number.any():
                    failure = 'physics_warning'
                    break
            report = dict(seed=seed, orientation=orientation, config=asdict(config),
                          success=failure is None, failure=failure, progress=progress,
                          seconds=float(sim.data.time), status=sim.status(),
                          max_drawer_opening=maximum_opening,
                          grasp_position=sim.data.site('grasp').xpos.tolist())
            if frames:
                video_path = args.output.with_name(f'{args.output.stem}_seed{seed}.gif')
                frames[0].save(video_path, save_all=True, append_images=frames[1:],
                               duration=100, loop=0)
                report['video'] = str(video_path)
            reports.append(report)
            summary.update(successes=sum(item['success'] for item in reports), trials=len(reports))
            args.output.write_text(json.dumps(summary, indent=2))
            trials.set_postfix(ok=f"{summary['successes']}/{summary['trials']}", refresh=False)
            if getattr(args, 'verbose', False):
                trials.write(json.dumps(report))
        finally:
            sim.close()
    print(f"Policy success: {summary['successes']}/{summary['trials']}; report: {args.output}")


def run() -> None:
    """Dispatch the collection, training, or evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collector = commands.add_parser('collect')
    collector.add_argument('--episodes', type=int, default=12)
    collector.add_argument('--seed-start', type=int, default=0)
    collector.add_argument('--randomization', choices=['none', 'cup', 'scene'], default='cup')
    collector.add_argument('--width', type=int, default=160)
    collector.add_argument('--height', type=int, default=120)
    collector.add_argument('--output', type=Path, required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--data', type=Path, required=True)
    trainer.add_argument('--output', type=Path, required=True)
    trainer.add_argument('--epochs', type=int, default=40)
    trainer.add_argument('--horizon', type=int, default=16)
    trainer.add_argument('--batch-size', type=int, default=64)
    trainer.add_argument('--lr', type=float, default=3e-4)
    trainer.add_argument('--seed', type=int, default=0)
    trainer.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    trainer.add_argument('--action-space', choices=['absolute', 'delta'], default='absolute')
    evaluator = commands.add_parser('evaluate')
    evaluator.add_argument('--checkpoint', type=Path, required=True)
    evaluator.add_argument('--episodes', type=int, default=6)
    evaluator.add_argument('--seed-start', type=int, default=1000)
    evaluator.add_argument('--randomization', choices=['cup', 'scene'], default='scene')
    evaluator.add_argument('--seconds', type=float, default=65)
    evaluator.add_argument('--execute', type=int, default=1)
    evaluator.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluator.add_argument('--allow-training-seeds', action='store_true')
    evaluator.add_argument('--video', action='store_true')
    evaluator.add_argument('--output', type=Path, required=True)
    for command in [collector, trainer, evaluator]:
        add_progress_arguments(command)
    args = parser.parse_args()
    for name in ['episodes', 'width', 'height', 'epochs', 'horizon', 'batch_size', 'execute', 'seconds', 'lr']:
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and positive.')
    dict(collect=collect, train=train, evaluate=evaluate)[args.command](args)


if __name__ == '__main__':
    run()
