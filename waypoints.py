"""RGBD waypoint imitation with a learned visual front end and robot IK execution."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from expert import ArmController, DOWN, FRONT, ExpertFailure
from fetch_assets import SOURCES
from learning import split_episodes
from progress import add_progress_arguments, progress_bar
from simulation import KitchenSimulation, ORIENTATIONS, SceneConfig


def world_points(observation:dict) -> np.ndarray:
    """Back-project calibrated RGBD to an H×W×3 world-coordinate image."""
    depth = np.minimum(observation['depth'].astype(np.float32), 3)
    v, u = np.indices(depth.shape, dtype=np.float32)
    calibration = observation['intrinsics']
    camera = np.stack([(u - calibration[0, 2]) / calibration[0, 0] * depth,
                       (v - calibration[1, 2]) / calibration[1, 1] * depth, depth], -1)
    transform = observation['camera_to_world']
    return (camera @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def expert_labels(sim:KitchenSimulation) -> tuple[np.ndarray, np.ndarray]:
    """Return training masks and a 14-value expert waypoint vector.

    Targets contain cup grasp XYZ, handle grasp XYZ, cup-center offset XYZ,
    cos/sin of twice the gripper yaw, and the cup's world vertical-axis direction.
    Twice-yaw represents the parallel gripper's symmetry under a half turn.
    """
    rotation = sim.data.body('cup').xmat.reshape(3, 3)
    grasp = sim.data.body('cup').xpos + rotation @ [0, 0, .037 * sim.config.cup_scale]
    axis = rotation[:, 1].copy()
    axis[2] = 0
    if np.linalg.norm(axis) < .5:
        axis = rotation[:, 0].copy()
        axis[2] = 0
        grasp = sim.data.body('cup').xpos + rotation @ [0, 0, .0026 * sim.config.cup_scale]
    axis /= np.linalg.norm(axis)
    angle = np.arctan2(axis[1], axis[0])
    minimum, maximum = sim.cup_bounds()
    offset = (minimum + maximum) / 2 - grasp
    name = dict(bar='handle_bar', knob='handle_knob', loop='handle_loop_middle')[sim.config.handle]
    handle = sim.data.geom(name).xpos.copy()
    target = np.r_[grasp, handle, offset, np.cos(2 * angle), np.sin(2 * angle), rotation[:, 2]]
    sim.renderer.enable_segmentation_rendering()
    try:
        segmentation = sim.renderer.render()
    finally:
        sim.renderer.disable_segmentation_rendering()
    labels = np.zeros(segmentation.shape[:2], np.uint8)
    geom = segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    ids = segmentation[..., 0]
    cup = np.flatnonzero(sim.model.geom_bodyid == sim.model.body('cup').id)
    handles = [index for index in range(sim.model.ngeom)
               if (mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, index) or '').startswith('handle_')]
    labels[geom & np.isin(ids, cup)] = 1
    labels[geom & np.isin(ids, handles)] = 2
    return labels, target.astype(np.float32)


def collect(args) -> None:
    """Save settled RGBD scenes with expert waypoint labels and training-only masks."""
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Choose a new dataset directory: {args.output}')
    manifest = dict(version=1, policy='rgbd_waypoint_imitation', randomization=args.randomization,
                    camera=args.camera,
                    asset_sources=SOURCES,
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ['simulation.py', 'expert.py', 'waypoints.py']},
                    mujoco_version=mujoco.__version__, episodes=[])
    sim = None
    scenes = progress_bar(range(args.episodes), desc='Collect', unit='scene',
                          disabled=getattr(args, 'no_progress', False))
    try:
        for index in scenes:
            seed = args.seed_start + index if args.randomization != 'none' else args.seed_start
            orientation = ORIENTATIONS[index % 3] if args.randomization != 'none' else 'upright'
            if sim is None or args.randomization == 'scene':
                if sim is not None:
                    sim.close()
                config = SceneConfig.randomized(seed) if args.randomization == 'scene' else SceneConfig()
                sim = KitchenSimulation(config, seed, orientation)
            else:
                sim.reset(seed, orientation)
            observation = sim.observe(args.width, args.height, args.camera)
            labels, target = expert_labels(sim)
            filename = f'scene_{index:05d}.npz'
            np.savez_compressed(args.output / filename, **observation, labels=labels, target=target)
            manifest['episodes'].append(dict(file=filename, seed=seed, orientation=orientation,
                                             config=asdict(config)))
            if (index + 1) % 25 == 0 or index == 0:
                manifest_path.write_text(json.dumps(manifest, indent=2))
            scenes.set_postfix_str(f'{seed} {orientation}, {config.handle}', refresh=False)
            if getattr(args, 'verbose', False):
                scenes.write(f'{index + 1}/{args.episodes}: seed={seed}, {orientation}, {config.handle}')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f'Saved {args.episodes} waypoint demonstrations to {args.output}', flush=True)
    finally:
        if sim is not None:
            sim.close()


class WaypointPolicy(nn.Module):
    def __init__(self):
        """Predict object masks, metric grasp offsets, cup orientation, and placement offset."""
        super().__init__()

        def block(inputs, outputs):
            return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())

        self.encoder1 = block(6, 24)
        self.encoder2 = block(24, 48)
        self.encoder3 = block(48, 96)
        self.encoder4 = block(96, 128)
        self.decoder3 = block(224, 96)
        self.decoder2 = block(144, 48)
        self.decoder1 = block(72, 24)
        self.output = nn.Conv2d(24, 17, 1)

    def forward(self, image):
        """Return class logits (3), grasp offsets (6), center offset (3), yaw (2), and axis (3)."""
        first = self.encoder1(image)
        second = self.encoder2(F.max_pool2d(first, 2))
        third = self.encoder3(F.max_pool2d(second, 2))
        fourth = self.encoder4(F.max_pool2d(third, 2))
        third = self.decoder3(torch.cat([F.interpolate(fourth, third.shape[-2:]), third], 1))
        second = self.decoder2(torch.cat([F.interpolate(third, second.shape[-2:]), second], 1))
        first = self.decoder1(torch.cat([F.interpolate(second, first.shape[-2:]), first], 1))
        return self.output(first)

    @staticmethod
    def decode(dense, xyz, masks=None):
        """Pool metric waypoint votes; inference uses predicted masks, never simulator masks."""
        if masks is None:
            probability = dense[:, :3].softmax(1)[:, 1:]
            masks = probability.pow(4) * (probability > .5)
        weights = masks / masks.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        points = xyz[:, None] + .1 * dense[:, 3:9].reshape(-1, 2, 3, *dense.shape[-2:])
        grasps = (points * weights[:, :, None]).sum((-2, -1)).flatten(1)
        cup = (dense[:, 9:] * weights[:, :1]).sum((-2, -1))
        return torch.cat([grasps, .1 * cup[:, :3], cup[:, 3:]], 1)


def prepare(observation:dict) -> tuple[np.ndarray, np.ndarray]:
    """Return network RGB/XYZ input and unnormalized world coordinates from RGBD only."""
    xyz = world_points(observation).transpose(2, 0, 1)
    rgb = observation['rgb'].astype(np.float32).transpose(2, 0, 1) / 255
    image = np.concatenate([rgb, xyz - np.array([0, 0, 1], np.float32)[:, None, None]])
    return image.astype(np.float32), xyz


class WaypointDemonstrations(Dataset):
    def __init__(self, directory:Path, entries:list[dict], show_progress:bool=False,
                 description:str='Loading waypoints'):
        """Load RGBD and expert waypoint supervision; metadata cannot become model inputs."""
        self.examples = []
        for entry in progress_bar(entries, desc=description, unit='scene',
                                  disabled=not show_progress, leave=False):
            with np.load(directory / entry['file'], allow_pickle=False) as source:
                image, xyz = prepare(source)
                self.examples.append((image, xyz, source['labels'], source['target']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class WaypointMemory:
    def __init__(self, points:torch.Tensor, targets:torch.Tensor):
        """Store training RGBD object clouds and their demonstrated metric waypoints."""
        self.centers = points.mean(2)
        self.points = points - self.centers[:, :, None]
        self.targets = targets

    @classmethod
    def from_demonstrations(cls, dataset:WaypointDemonstrations, device:str, show_progress:bool=False):
        """Build a nonparametric imitation model from the training partition alone."""
        points, targets = [], []
        seen = set()
        for image, xyz, labels, target in progress_bar(dataset.examples, desc='Build memory',
                unit='scene', disabled=not show_progress, leave=False):
            if target.tobytes() in seen:
                continue
            seen.add(target.tobytes())
            clouds = [xyz[:, labels == label].T for label in [1, 2]]
            points.append(np.stack([cloud[np.linspace(0, len(cloud) - 1, 128).astype(int)]
                                    for cloud in clouds]))
            targets.append(target)
        return cls(torch.tensor(np.array(points), device=device),
                   torch.tensor(np.array(targets), device=device))

    def state(self) -> dict:
        return dict(points=(self.points + self.centers[:, :, None]).cpu(),
                    targets=self.targets.cpu())

    def register(self, cloud:torch.Tensor, object_index:int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Match observed 3D shape to demonstrations and refine its yaw/translation with ICP."""
        center = cloud.mean(0)
        cloud = cloud - center
        source = self.points[:, object_index]
        distance = torch.cdist(source, cloud[None])
        score = distance.min(-1).values.mean(-1) + distance.min(-2).values.mean(-1)
        indices = score.topk(min(12, len(source)), largest=False).indices
        source = source[indices]
        rotation = torch.eye(3, device=cloud.device).expand(len(source), -1, -1).clone()
        translation = torch.zeros(len(source), 3, device=cloud.device)
        for _ in range(15):
            transformed = source @ rotation.transpose(1, 2) + translation[:, None]
            distance = torch.cdist(transformed, cloud[None])
            forward = distance.argmin(-1)
            backward = distance.argmin(-2)
            paired_source = torch.cat([source, source.gather(1, backward[..., None].expand(-1, -1, 3))], 1)
            paired_target = torch.cat([cloud[forward], cloud[None].expand(len(source), -1, -1)], 1)
            source_mean, target_mean = paired_source.mean(1), paired_target.mean(1)
            a, b = paired_source - source_mean[:, None], paired_target - target_mean[:, None]
            if object_index == 0:
                cross = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(1)
                dot = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(1)
                angle = torch.atan2(cross, dot).clamp(-.35, .35)
                c, s = angle.cos(), angle.sin()
                rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
                rotation[:, 1, 0], rotation[:, 1, 1] = s, c
            translation = target_mean - (source_mean[:, None] @ rotation.transpose(1, 2))[:, 0]
        distance = torch.cdist(source @ rotation.transpose(1, 2) + translation[:, None], cloud[None])
        score = distance.min(-1).values.mean(-1) + distance.min(-2).values.mean(-1)
        best = score.argmin()
        index = int(indices[best])
        transform = center + translation[best] - rotation[best] @ self.centers[index, object_index]
        return index, rotation[best], transform

    def predict(self, dense:torch.Tensor, xyz:torch.Tensor, cup_only:bool=False,
                support_height:float|None=None) -> np.ndarray:
        """Transfer demonstrated waypoints using learned masks and current RGBD geometry."""
        probability = dense[0, :3].softmax(0)
        matches = []
        for object_index in range(1 if cup_only else 2):
            mask = probability[object_index + 1] > (.35 if object_index == 0 else .7)
            if object_index == 0 and support_height is not None:
                # Remove tabletop pixels at uncertain cup boundaries using observed depth.
                mask &= (xyz[0, 2] > support_height + .001) & (xyz[0, 2] < support_height + .14)
            cloud = xyz[0, :, mask].T
            if len(cloud) < 8:
                raise ExpertFailure('Cup or drawer handle not visible to the learned model')
            cloud = cloud[torch.linspace(0, len(cloud) - 1, 128, device=cloud.device).long()]
            matches.append(self.register(cloud, object_index))
        index, rotation, translation = matches[0]
        target = self.targets[index].clone()
        target[:3] = rotation @ target[:3] + translation
        target[6:9] = rotation @ target[6:9]
        angle = torch.atan2(target[10], target[9]) / 2 + torch.atan2(rotation[1, 0], rotation[0, 0])
        target[9], target[10] = (2 * angle).cos(), (2 * angle).sin()
        target[11:14] = rotation @ target[11:14]
        if not cup_only:
            index, rotation, translation = matches[1]
            target[3:6] = rotation @ self.targets[index, 3:6] + translation
        return target.cpu().numpy()


def loss_function(dense, xyz, labels, target):
    """Supervise visual masks and expert metric waypoints, with dense offset consistency."""
    masks = torch.stack([labels == 1, labels == 2], 1).float()
    weights = masks / masks.sum((-2, -1), keepdim=True).clamp_min(1)
    predicted = WaypointPolicy.decode(dense, xyz, masks)
    classes = F.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2)
    loss = -(dense[:, :3].log_softmax(1) * classes).sum((-2, -1))
    loss = (loss / classes.sum((-2, -1)).clamp_min(1)).mean()
    probability = dense[:, :3].softmax(1)[:, 1:]
    loss += (1 - (2 * (probability * masks).sum((-2, -1)) + 1) /
             (probability.sum((-2, -1)) + masks.sum((-2, -1)) + 1)).mean()
    offsets = (target[:, :6].reshape(-1, 2, 3, 1, 1) - xyz[:, None]) / .1
    error = dense[:, 3:9].reshape_as(offsets) - offsets
    loss += (error.square() * weights[:, :, None]).sum((-2, -1)).mean()
    cup_target = torch.cat([target[:, 6:9] / .1, target[:, 9:]], 1)
    loss += ((dense[:, 9:] - cup_target[:, :, None, None]).square()
             * weights[:, :1]).sum((-2, -1)).mean()
    loss += F.mse_loss(predicted[:, :9] / .1, target[:, :9] / .1)
    return loss


def train(args) -> None:
    """Train waypoint imitation on independent scene splits and save the best validation model."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    training, validation = split_episodes(manifest['episodes'], args.seed)
    no_progress = getattr(args, 'no_progress', False)
    datasets = [WaypointDemonstrations(args.data, entries, not no_progress, f'Load {name}')
                for name, entries in [('training', training), ('validation', validation)]]
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = WaypointPolicy().to(device)
    memory = WaypointMemory.from_demonstrations(datasets[0], device, not no_progress).state()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    training_sampler = (RandomSampler(datasets[0], replacement=True,
                                     num_samples=args.steps_per_epoch * args.batch_size)
                        if args.steps_per_epoch else None)
    loaders = [DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=index == 0 and training_sampler is None,
                          sampler=training_sampler if index == 0 else None)
               for index, dataset in enumerate(datasets) if len(dataset)]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 10)
    history, best = [], float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Training on {device}: {len(training)} training / {len(validation)} validation scenes', flush=True)
    epochs = progress_bar(range(args.epochs), desc='Epochs', unit='epoch', disabled=no_progress)
    for epoch in epochs:
        losses, errors = [], []
        for index, loader in enumerate(loaders):
            model.train(index == 0)
            total, metric, count = 0., np.zeros(3), 0
            phase = 'Train' if index == 0 else 'Validate'
            with progress_bar(loader, desc=f'{phase} {epoch + 1}/{args.epochs}', unit='batch',
                              disabled=no_progress, leave=False, position=1) as batches:
                for image, xyz, labels, target in batches:
                    image, xyz, labels, target = [array.to(device) for array in [image, xyz, labels, target]]
                    with torch.set_grad_enabled(index == 0):
                        dense = model(image)
                        loss = loss_function(dense, xyz, labels, target)
                        if index == 0:
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 5)
                            optimizer.step()
                        prediction = model.decode(dense, xyz)
                        distances = torch.linalg.vector_norm((prediction[:, :9] - target[:, :9]).reshape(-1, 3, 3), dim=2)
                        metric += distances.detach().sum(0).cpu().numpy()
                    count += len(image)
                    total += float(loss.detach()) * len(image)
                    batches.set_postfix(loss=f'{total / count:.4f}', refresh=False)
            losses.append(total / count)
            errors.append((metric / count * 1000).tolist())
        history.append(dict(epoch=epoch + 1, train_loss=losses[0],
                            validation_loss=losses[-1] if validation else None,
                            train_error_mm=errors[0], validation_error_mm=errors[-1] if validation else None))
        if losses[-1] < best:
            best = losses[-1]
            checkpoint = dict(version=1, kind='rgbd_waypoint_imitation', state=model.state_dict(),
                              image_shape=list(datasets[0][0][0].shape[-2:]),
                              training_episodes=training, validation_episodes=validation,
                              randomization=manifest['randomization'], seed=args.seed,
                              camera=manifest.get('camera', 'rgbd'),
                              memory=memory, training_settings=dict(epochs=args.epochs,
                                  steps_per_epoch=len(loaders[0]), batch_size=args.batch_size, lr=args.lr),
                              checkpoint_selection='validation_loss' if validation else 'training_loss',
                              input_keys=['rgb', 'depth', 'camera_calibration', 'robot_state'], history=history)
            temporary = args.output.with_suffix('.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
        epochs.set_postfix(train=f'{losses[0]:.4f}', val=f'{losses[-1]:.4f}' if validation else 'n/a',
                           refresh=False)
        if getattr(args, 'verbose', False):
            epochs.write(json.dumps(history[-1]))
        scheduler.step()
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2))
    print(f'Saved best checkpoint to {args.output}; history: {args.output.with_suffix(".history.json")}', flush=True)


def execute(controller:ArmController, target:np.ndarray, height:float, perceive=None) -> None:
    """Execute predicted waypoints with robot feedback; no object-state queries or fallback."""
    grasp, closed_handle, offset = target[:9].reshape(3, 3).copy()
    handle = closed_handle.copy()
    move, hold = controller.move, controller.hold
    move(handle + [0, -.18, .23], FRONT, seconds=3, stage='approach_drawer')
    move(handle + [0, -.14, 0], FRONT, stage='lower_to_handle')
    move(handle, FRONT, stage='reach_handle')
    hold(.8, 0)
    handle = closed_handle + [0, -.34, 0]
    move(handle, FRONT, 0, 4, 'open_drawer')
    hold(.5, 255)
    move(controller.sim.data.site('grasp').xpos + [0, -.04, 0], FRONT, stage='release_handle')
    move(controller.sim.data.site('grasp').xpos + [0, 0, .40], DOWN, stage='clear_drawer')
    for attempt, turn in enumerate([0, np.pi / 2, np.pi / 4]):
        angle = np.arctan2(target[10], target[9]) / 2 + turn
        axis = np.array([np.cos(angle), np.sin(angle), 0])
        if axis[1] > 0:
            axis *= -1
        down = np.column_stack([np.cross(axis, [0, 0, -1]), axis, [0, 0, -1]])
        grasp, offset = target[:3].copy(), target[6:9].copy()
        try:
            move(np.r_[grasp[:2], height + .25], down, seconds=3, stage='approach_cup')
            move(grasp, down, stage='reach_cup')
            controller.stage = 'grasp_cup'
            hold(.8, 0)
            if controller.sim.data.qpos[7:9].sum() < .012:
                raise ExpertFailure('Empty gripper after cup grasp')
            move(grasp + [0, 0, .22], down, 0, 2, 'lift_cup')
            if controller.sim.data.qpos[7:9].sum() < .012:
                raise ExpertFailure('Cup slipped from gripper during lift')
            break
        except ExpertFailure:
            if attempt == 2:
                raise
            hold(.3, 255)
            move(controller.sim.data.site('grasp').xpos + [0, 0, .12], down,
                 stage='retreat_for_retry')
            move([closed_handle[0], closed_handle[1] - .38, height + .25], DOWN,
                 stage='clear_for_retry')
            if perceive is not None:
                target = perceive()
    place = handle + [0, .20, .02] - offset
    move(np.r_[place[:2], height + .25], down, 0, 3, 'carry_cup')
    move(place, down, 0, 3, 'lower_cup')
    hold(.8, 255)
    move(place + [0, 0, .30], down, stage='retreat_from_cup')
    move(handle + [0, -.08, .22], FRONT, seconds=3, stage='approach_closing')
    move(handle + [0, -.06, 0], FRONT, stage='lower_for_closing')
    move(handle, FRONT, stage='reach_for_closing')
    hold(.8, 0)
    move(closed_handle + [0, .015, 0], FRONT, 0, 4, 'close_drawer')
    hold(.5, 255)
    move(closed_handle + [0, -.13, 0], FRONT, stage='release_drawer')
    hold(.5, 255)


def countertop_height(xyz:np.ndarray) -> float:
    """Estimate the largest horizontal surface height from the depth point cloud."""
    points = xyz.reshape(-1, 3)
    z = points[(points[:, 1] > .07) & (np.abs(points[:, 0]) < .5)
               & (points[:, 2] > .6) & (points[:, 2] < 1.05), 2]
    bins = np.round(z / .002).astype(int)
    if not len(bins):
        raise ExpertFailure('No visible countertop in RGBD')
    mode = np.bincount(bins).argmax() * .002
    return float(np.median(z[np.abs(z - mode) < .003]))


def evaluate(args) -> None:
    """Run the visual waypoint policy on disjoint seeds and score only outside the controller."""
    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WaypointPolicy().to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    memory = (WaypointMemory(checkpoint['memory']['points'].to(device),
                             checkpoint['memory']['targets'].to(device))
              if args.decoder == 'registered' and 'memory' in checkpoint else None)
    used = {entry['seed'] for key in ['training_episodes', 'validation_episodes'] for entry in checkpoint[key]}
    requested = set(range(args.seed_start, args.seed_start + args.episodes))
    if requested & used and not args.allow_training_seeds:
        raise ValueError('Evaluation seeds overlap development data.')
    summary = dict(policy=checkpoint['kind'], checkpoint=str(args.checkpoint),
                   randomization=args.randomization, seed_overlap=sorted(requested & used),
                   training_randomization=checkpoint['randomization'],
                   decoder='registered' if memory is not None else 'neural',
                   checkpoint_epoch=checkpoint['history'][-1]['epoch'],
                   training_settings=checkpoint.get('training_settings'),
                   mujoco_version=mujoco.__version__,
                   source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ['simulation.py', 'expert.py', 'waypoints.py']},
                   successes=0, trials=0, episodes=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trials = progress_bar(range(args.episodes), desc='Eval', unit='trial',
                          disabled=getattr(args, 'no_progress', False))
    for index in trials:
        seed, orientation = args.seed_start + index, ORIENTATIONS[index % 3]
        trials.set_postfix_str(f'{seed} reset | {summary["successes"]} ok')
        config = SceneConfig.randomized(seed) if args.randomization == 'scene' else SceneConfig()
        sim = KitchenSimulation(config, seed, orientation)
        progress = dict(drawer_opened=False, cup_lifted=False, cup_placed=False, drawer_closed=False)
        frames, count, max_opening, last_stage = [], 0, 0., None

        def measure(simulation, control, stage):
            """Measure task order and optionally capture video, without feeding metrics to control."""
            nonlocal count, max_opening, last_stage
            if not trials.disable and stage != last_stage:
                trials.set_postfix_str(f'{seed} {stage.replace("_", " ")} | {summary["successes"]} ok')
                last_stage = stage
            status = simulation.status()
            max_opening = max(max_opening, status['drawer_opening'])
            progress['drawer_opened'] |= status['drawer_opening'] > .28
            minimum, _ = simulation.cup_bounds()
            progress['cup_lifted'] |= bool(progress['drawer_opened'] and minimum[2] > config.counter_height + .06)
            progress['cup_placed'] |= progress['cup_lifted'] and status['cup_in_drawer']
            progress['drawer_closed'] |= progress['cup_placed'] and status['success']
            if args.video and count % 5 == 0:
                from PIL import Image
                frames.append(Image.fromarray(simulation.observe(320, 240)['rgb']))
            count += 1

        def perceive(cup_only=False):
            """Estimate waypoints from the current RGBD frame, including after a failed grasp."""
            height, width = checkpoint['image_shape']
            observation = sim.observe(width, height, checkpoint.get('camera', 'rgbd'))
            image, xyz = prepare(observation)
            with torch.inference_mode():
                dense = model(torch.from_numpy(image[None]).to(device))
                points = torch.from_numpy(xyz[None]).to(device)
                surface = countertop_height(xyz.transpose(1, 2, 0))
                target = (memory.predict(dense, points, cup_only, surface) if memory is not None
                          else model.decode(dense, points)[0].cpu().numpy())
            return target, xyz

        try:
            controller = ArmController(sim, measure)
            failure, target, surface = None, None, None
            try:
                target, xyz = perceive()
                surface = countertop_height(xyz.transpose(1, 2, 0))
                if not np.isfinite(target).all():
                    raise ExpertFailure('Non-finite predicted waypoint')
                execute(controller, target, surface, lambda: perceive(cup_only=True)[0])
            except ExpertFailure as error:
                failure = str(error)
            measure(sim, None, 'complete')
            success = bool(all(progress.values()) and sim.status()['success'] and sim.is_stable())
            report = dict(seed=seed, orientation=orientation, config=asdict(config), success=success,
                          failure=failure if failure else None if success else 'Final task check failed',
                          progress=progress, seconds=float(sim.data.time), status=sim.status(),
                          stage=controller.stage, predicted_waypoints=target.tolist() if target is not None else None,
                          estimated_counter_height=surface, max_drawer_opening=max_opening)
            if frames:
                trials.set_postfix_str(f'{seed} save video | {summary["successes"]} ok')
                video = args.output.with_name(f'{args.output.stem}_seed{seed}.gif')
                frames[0].save(video, save_all=True, append_images=frames[1:], duration=100, loop=0)
                report['video'] = str(video)
            summary['episodes'].append(report)
            summary.update(trials=len(summary['episodes']),
                           successes=sum(item['success'] for item in summary['episodes']))
            args.output.write_text(json.dumps(summary, indent=2))
            trials.set_postfix_str(f"{summary['successes']} ok, {summary['trials'] - summary['successes']} failed",
                                   refresh=False)
            if getattr(args, 'verbose', False):
                trials.write(json.dumps(report))
        finally:
            sim.close()
    print(f"Waypoint policy success: {summary['successes']}/{summary['trials']}; report: {args.output}", flush=True)


def compare(args) -> None:
    """Compare paired held-out scenes only when training budgets and evaluation conditions match."""
    paths = dict(none=args.original_report, cup=args.cup_report, scene=args.scene_report)
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    reference = reports['scene']
    keys = lambda report: [(item['seed'], item['orientation'], json.dumps(item['config'], sort_keys=True))
                           for item in report['episodes']]
    for name, report in reports.items():
        if (keys(report) != keys(reference) or report['seed_overlap']
                or report['training_settings'] != reference['training_settings']
                or report['decoder'] != reference['decoder']
                or report['training_randomization'] != name):
            raise ValueError('Comparison requires paired scenes, disjoint seeds, and matched training settings.')
    results = {}
    for name, report in reports.items():
        count = sum(item['success'] for item in report['episodes'])
        trials = len(report['episodes'])
        rate, z = count / trials, 1.96
        center = (rate + z ** 2 / (2 * trials)) / (1 + z ** 2 / trials)
        radius = z * np.sqrt(rate * (1 - rate) / trials + z ** 2 / (4 * trials ** 2)) / (1 + z ** 2 / trials)
        groups = {}
        for field in ['orientation', 'handle']:
            groups[field] = {}
            for episode in report['episodes']:
                key = episode['orientation'] if field == 'orientation' else episode['config']['handle']
                group = groups[field].setdefault(key, dict(successes=0, trials=0))
                group['successes'] += int(episode['success'])
                group['trials'] += 1
        results[name] = dict(report=str(paths[name]), checkpoint=report['checkpoint'],
                             successes=count, trials=trials, success_rate=rate,
                             wilson_95=[max(0., float(center - radius)), min(1., float(center + radius))], **groups)
        print(f'{name}: {count}/{trials} ({rate:.1%})', flush=True)
    original, varied = reports['none']['episodes'], reference['episodes']
    summary = dict(scope='simulation_only', training_settings=reference['training_settings'],
                   seed_start=min(item['seed'] for item in varied), trials=len(varied), results=results,
                   paired_scene_vs_original=dict(
                       improved=sum(b['success'] and not a['success'] for a, b in zip(original, varied)),
                       regressed=sum(a['success'] and not b['success'] for a, b in zip(original, varied))),
                   limitations=['One training seed per condition; small evaluation set.',
                                'Fixed-scene checkpoint selection uses training loss; others use validation loss.',
                                'Known assets and ideal RGBD; no real-world transfer measurement.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


def run() -> None:
    """Collect expert waypoint demonstrations, train visual imitation, or evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collector = commands.add_parser('collect')
    collector.add_argument('--episodes', type=int, default=512)
    collector.add_argument('--seed-start', type=int, default=2000)
    collector.add_argument('--randomization', choices=['none', 'cup', 'scene'], default='scene')
    collector.add_argument('--width', type=int, default=384)
    collector.add_argument('--height', type=int, default=288)
    collector.add_argument('--camera', choices=['rgbd', 'workspace_rgbd'], default='workspace_rgbd')
    collector.add_argument('--output', type=Path, required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--data', type=Path, required=True)
    trainer.add_argument('--epochs', type=int, default=60)
    trainer.add_argument('--batch-size', type=int, default=16)
    trainer.add_argument('--lr', type=float, default=3e-4)
    trainer.add_argument('--seed', type=int, default=0)
    trainer.add_argument('--steps-per-epoch', type=int)
    trainer.add_argument('--output', type=Path, required=True)
    trainer.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluator = commands.add_parser('evaluate')
    evaluator.add_argument('--checkpoint', type=Path, required=True)
    evaluator.add_argument('--episodes', type=int, default=6)
    evaluator.add_argument('--seed-start', type=int, default=1000)
    evaluator.add_argument('--randomization', choices=['cup', 'scene'], default='scene')
    evaluator.add_argument('--output', type=Path, required=True)
    evaluator.add_argument('--video', action='store_true')
    evaluator.add_argument('--allow-training-seeds', action='store_true')
    evaluator.add_argument('--decoder', choices=['registered', 'neural'], default='registered')
    evaluator.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    comparison = commands.add_parser('compare')
    comparison.add_argument('--original-report', type=Path, required=True)
    comparison.add_argument('--cup-report', type=Path, required=True)
    comparison.add_argument('--scene-report', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    for command in [collector, trainer, evaluator]:
        add_progress_arguments(command)
    args = parser.parse_args()
    for name in ['episodes', 'width', 'height', 'epochs', 'batch_size', 'steps_per_epoch', 'lr']:
        value = getattr(args, name, None)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and positive.')
    dict(collect=collect, train=train, evaluate=evaluate, compare=compare)[args.command](args)


if __name__ == '__main__':
    run()
