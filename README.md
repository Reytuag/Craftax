# 📜 Basic Usage
Craftax conforms to the gymnax interface:
```python
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
env = AutoResetEnvWrapper(CraftaxSymbolicEnv())
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(rngs[1])

# Step environment
obs, state, reward, done, info = env.step(rngs[2], state, action, env_params)
```

## Extending Craftax
If you want to extend Craftax, run (make sure you have `pip>=23.0`):
```
git clone ...
cd Craftax
pip install --editable .
```

## GPU-Enabled JAX
By default, both of the above methods will install JAX on the CPU.  If you want to run JAX on a GPU/TPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# 🎮 Play
To play Craftax run:
```
play_craftax
```
or to play Craftax-Classic run:
```
play_craftax_classic
```
Since Craftax runs entirely in JAX, it will take some time to compile the rendering and step functions - it might take around 30s to render the first frame and then another 20s to take the first action.  After this it should be very quick.  A tutorial for how to beat the game is present in `tutorial.md`.  The controls are printed out at the beginning of play.

# 📈 Experiment
To run PPO with default hyperparameters run:
```
python -m craftax.ppo
```
or to run PPO with memory call:
```
python -m craftax.ppo_rnn
```
To use ICM or E3B with the default parameters use the `--train_icm` and `--use_e3b` flags.
Use the `env_name` parameter to control which environment is used.  It can be set to  `"Craftax-Symbolic-v1"`, `"Craftax-Pixels-v1"`, `"Craftax-Classic-Symbolic-v1"` or `"Craftax-Classic-Pixels-v1"`

# 🔪 Gotchas
### Optimistic Resets
Craftax provides the option to use optimistic resets to improve performance, which means that (unlike regular gymnax environments) it **does not auto-reset** by default.
This means that the environment should always be wrapped either in `OptimisticResetVecEnvWrapper` (for efficient resets) or `AutoResetEnvWrapper` (to recover the default gymnax auto-reset behaviour).  See `ppo.py` for correct usage of both wrappers.

### Texture Caching
We use a texture cache to avoid recreating the texture atlas every time Craftax is imported. If you are just running Craftax as a benchmark this will not affect you.  However, if you are editing the game (e.g. adding new blocks, entities etc.) then a stale cache could cause errors. You can export the following environment variable to force textures to be created from scratch.
```
export CRAFTAX_RELOAD_TEXTURES=true
```