from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .agent import TGODSACAgent
from .checkpoint_config import assert_checkpoint_compatible
from .config import (PROJECT_ROOT, resolve_input_path, resolve_output_path, select_device,
                     validate_reproduction_training)
from .env import UR5ePickPlaceEnv
from .evaluation import EpisodeDiagnostics
from .expert import ExpertTrajectory
from .replay_buffer import ReplayBuffer
from .schema import ACTION_DIM, OBS_DIM
from .trajectory import generate_and_match, one_hot_skill


TRAINING_PROTOCOL = 'equal_mine_fixed_budget_v1'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_components(config: dict[str, Any], *, render_mode: str | None = None):
    scene = resolve_input_path(config['paths']['scene_xml'], kind='scene')
    expert = ExpertTrajectory.load(resolve_input_path(config['paths']['expert_dir'], kind='expert'))
    device = select_device(str(config['device']))
    env = UR5ePickPlaceEnv(scene, expert, config['environment'], render_mode=render_mode)
    agent = TGODSACAgent(OBS_DIM, ACTION_DIM, expert.relation_dim, config, device)
    return expert, env, agent, resolve_output_path(config['paths']['output_dir']), device


def save_checkpoint(path: Path, *, agent: TGODSACAgent, config: dict[str, Any],
                    episode: int, global_step: int, rng: np.random.Generator,
                    replay: ReplayBuffer | None = None,
                    training_state: dict[str, Any] | None = None,
                    provenance: dict[str, Any] | None = None,
                    action_rng_state: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        'format_version': 1, 'training_protocol': TRAINING_PROTOCOL,
        'agent': agent.state_dict(), 'config': config,
        'episode': int(episode), 'global_step': int(global_step),
        'numpy_rng_state': rng.bit_generator.state,
        'python_rng_state': random.getstate(), 'numpy_global_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'training_state': training_state or {}, 'provenance': provenance or {},
        'action_rng_state': action_rng_state,
    }
    if replay is not None:
        document['replay_buffer'] = replay.state_dict()
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(document, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, agent: TGODSACAgent, *, load_optimizers: bool) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=agent.device, weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    agent.load_state_dict(checkpoint['agent'], load_optimizers=load_optimizers)
    return checkpoint


def _restore_rng(checkpoint: dict[str, Any], rng: np.random.Generator) -> None:
    if 'numpy_rng_state' in checkpoint:
        rng.bit_generator.state = checkpoint['numpy_rng_state']
    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])
    if 'numpy_global_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_global_rng_state'])
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_state_all']])


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]} if metrics else {}


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _append_json(path: Path, document: Any) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(document, ensure_ascii=False, allow_nan=False) + '\n')


def _provenance(config: dict[str, Any]) -> dict[str, Any]:
    scene = resolve_input_path(config['paths']['scene_xml'], kind='scene')
    expert_dir = resolve_input_path(config['paths']['expert_dir'], kind='expert')
    paths = list(scene.parent.rglob('*.xml')) + [expert_dir / name for name in (
        'expert_demo.npy', 'expert_qpos.npy', 'expert_cup.npy', 'expert_initial_state.npz')]
    assets = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    source_files = list((PROJECT_ROOT / 'tgod_sd').glob('*.py')) + [PROJECT_ROOT / 'train.py', PROJECT_ROOT / 'evaluate.py']
    sources = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'git_commit': commit, 'git_dirty': dirty, 'asset_sha256': assets, 'source_sha256': sources}


def _validate_resume(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    saved = checkpoint.get('config', {})
    if checkpoint.get('training_protocol') != TRAINING_PROTOCOL:
        raise ValueError('Checkpoint predates the equal-MINE fixed-budget protocol. '
                         'Start a fresh run with configs/paper_seed45.yaml; historical policies remain evaluable.')
    validate_reproduction_training(saved)
    assert_checkpoint_compatible(saved, config)
    # A resumed reproduction run continues the same experiment, including its seed
    # and optimizer settings. Changes belong in a separate, explicitly named run.
    for section in ('seed', 'tgod', 'sac'):
        if saved.get(section) != config.get(section):
            raise ValueError(f'Cannot change {section} during reproduction resume; start a fresh run.')
    replay_state = checkpoint.get('replay_buffer')
    legacy_metadata = isinstance(replay_state, dict) and set(replay_state) == {'capacity', 'index', 'size'}
    if replay_state is None or legacy_metadata:
        raise ValueError('Resume requires a complete replay buffer. Use the new run\'s latest.pt; '
                         'lightweight or historical checkpoints can only be evaluated.')
    if checkpoint.get('training_state', {}).get('interrupted'):
        raise ValueError('Checkpoint was saved inside an incomplete episode; use a completed-episode checkpoint.')
    for key in ('numpy_rng_state', 'python_rng_state', 'numpy_global_rng_state',
                'torch_rng_state', 'action_rng_state'):
        if checkpoint.get(key) is None:
            raise ValueError(f'Resume checkpoint is missing {key}; cannot continue the same sampling sequence.')


def _add_transition(replay, expert, observation, action, next_observation, skill,
                    terminated, truncated, sac) -> None:
    replay.add(observation, action, next_observation, skill,
               expert.relation_feature(observation, float(observation[-1])),
               expert.relation_feature(next_observation, float(next_observation[-1])),
               terminal=bool(terminated or (truncated and not sac['bootstrap_on_timeout'])))


def train(config: dict[str, Any], resume_path: str | Path | None = None, *,
          additional_episodes: int | None = None) -> Path:
    """Train to a declared episode budget; task diagnostics never select or stop a model."""
    config = copy.deepcopy(config)
    validate_reproduction_training(config)
    if additional_episodes is not None and (
        isinstance(additional_episodes, bool) or int(additional_episodes) != additional_episodes
        or additional_episodes <= 0 or not resume_path
    ):
        raise ValueError('--additional-episodes must be positive and requires a resume checkpoint.')
    if resume_path and not Path(resume_path).expanduser().is_file():
        raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
    seed = int(config['seed'])
    seed_everything(seed)
    expert, env, agent, output_directory, device = build_components(config)
    try:
        checkpoints_directory = output_directory / 'checkpoints'
        metrics_path = output_directory / 'metrics.jsonl'
        sac, training = config['sac'], config['training']
        replay = ReplayBuffer(int(sac['replay_size']), OBS_DIM, ACTION_DIM, agent.skill_dim, expert.relation_dim, seed)
        rng = np.random.default_rng(seed)
        env.action_space.seed(seed)
        start_episode = global_step = 0
        checkpoint: dict[str, Any] = {}
        if resume_path:
            checkpoint = torch.load(Path(resume_path).expanduser(), map_location=agent.device, weights_only=False)
            if checkpoint.get('format_version') != 1:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
            _validate_resume(checkpoint, config)
            agent.load_state_dict(checkpoint['agent'], load_optimizers=True)
            agent.apply_learning_rates(config)
            replay.load_state_dict(checkpoint['replay_buffer'])
            start_episode, global_step = int(checkpoint['episode']), int(checkpoint['global_step'])
            _restore_rng(checkpoint, rng)
            env.action_space.np_random.bit_generator.state = checkpoint['action_rng_state']
            print(f'Resumed episode={start_episode}, global_step={global_step}, replay_restored=True', flush=True)
        total_episodes = start_episode + int(additional_episodes) if additional_episodes is not None else int(training['episodes'])
        training['episodes'] = total_episodes
        if start_episode > total_episodes:
            raise ValueError(f'Checkpoint is at episode {start_episode}, beyond total {total_episodes}; increase --episodes.')
        if metrics_path.exists() and metrics_path.stat().st_size:
            with metrics_path.open(encoding='utf-8') as handle:
                last_logged = max(json.loads(line)['episode'] for line in handle if line.strip())
            if not resume_path or last_logged > start_episode:
                raise ValueError('Output contains later training episodes. Use a new --output-dir to avoid mixing experiments.')
        provenance = _provenance(config)
        saved_assets = checkpoint.get('provenance', {}).get('asset_sha256')
        if saved_assets and sorted(saved_assets.values()) != sorted(provenance['asset_sha256'].values()):
            raise ValueError('Scene or demonstration assets differ from the checkpoint; use a separate fresh experiment.')
        del checkpoint
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / 'config.resolved.yaml').write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        manifest = {'training_protocol': TRAINING_PROTOCOL,
                    'resume_checkpoint': str(Path(resume_path).resolve()) if resume_path else None,
                    'start_episode': start_episode, 'total_episodes': total_episodes,
                    'optimizer_learning_rates': agent.optimizer_learning_rates(),
                    'replay_restored': bool(resume_path), 'config': config, **provenance}
        _write_json(output_directory / 'run_manifest.json', manifest)
        print(f'TGOD-SD on {device}: episodes={start_episode}->{total_episodes}, skills={agent.skill_dim}', flush=True)
        print(f'Actual optimizer learning rates: {agent.optimizer_learning_rates()}', flush=True)
        latest_path = checkpoints_directory / 'latest.pt'
        completed_episode = start_episode

        def save(path: Path, *, full: bool = False) -> None:
            save_checkpoint(path, agent=agent, config=config, episode=completed_episode,
                            global_step=global_step, rng=rng, replay=replay if full else None,
                            provenance=provenance,
                            action_rng_state=env.action_space.np_random.bit_generator.state)

        # Only episode-boundary states are resumable; never overwrite these with
        # a learner/replay captured halfway through an interrupted episode.
        save(latest_path, full=True)
        try:
            for episode in range(start_episode, total_episodes):
                skill_index = int(rng.integers(agent.skill_dim))
                skill = one_hot_skill(skill_index, agent.skill_dim)
                observation, reset_info = env.reset(seed=seed + episode)
                diagnostics = EpisodeDiagnostics(env.success_radius, env.success_z_max)
                diagnostics.update(None, reset_info)
                updates: list[dict[str, float]] = []
                while True:
                    action = env.action_space.sample() if global_step < int(sac['random_steps']) else agent.act(observation, skill, deterministic=False)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    if reward != 0.0:
                        raise RuntimeError('TGOD environment reward must remain zero; MINE supplies the pseudo-reward.')
                    _add_transition(replay, expert, observation, action, next_observation, skill, terminated, truncated, sac)
                    observation = next_observation
                    global_step += 1
                    diagnostics.update(action, info)
                    if len(replay) >= max(int(sac['batch_size']), int(sac['update_after'])) and global_step % int(sac['update_every']) == 0:
                        for _ in range(int(sac['gradient_steps'])):
                            updates.append(agent.update(replay.sample(int(sac['batch_size']))))
                    if terminated or truncated:
                        break
                completed_episode = episode + 1
                stages, update_metrics = diagnostics.as_dict(), _mean_metrics(updates)
                _append_json(metrics_path, {'episode': completed_episode, 'global_step': global_step,
                                           'skill_index': skill_index, 'replay_size': len(replay), **stages, **update_metrics})
                if completed_episode % int(training['log_every_episodes']) == 0 or episode == start_episode:
                    print(f"Episode {completed_episode}/{total_episodes}: success={stages['success']}, skill={skill_index}, "
                          f"grasp={stages['ever_grasped']}, lift={stages['cup_lifted']}, "
                          f"distance={stages['final_goal_distance']:.4f}, pseudo={update_metrics.get('pseudo_reward_raw_mean', float('nan')):.4f}", flush=True)
                if completed_episode % int(training['checkpoint_every_episodes']) == 0:
                    save(checkpoints_directory / f'episode_{completed_episode:05d}.pt')
                    save(latest_path, full=True)
        except KeyboardInterrupt:
            print(f'Interrupted. The last completed checkpoint is preserved at {latest_path}; '
                  'use a new output directory if later episode logs already exist.', flush=True)
            raise
        save(latest_path, full=True)
        if bool(training['match_after_training']):
            generate_and_match(env, expert, agent, config['matching'], output_directory, seed=seed)
        print(f'Training complete at episode {completed_episode}. Latest: {latest_path}', flush=True)
        return latest_path
    finally:
        env.close()
