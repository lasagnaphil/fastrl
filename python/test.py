import fastrl_py as rl
import torch
import gym

policy_opt = rl.PolicyOptions()
policy_opt.actor_hidden_dim = [64, 64]
policy_opt.critic_hidden_dim = [64, 64]
policy_opt.activation_type = rl.NNActivationType.ReLU

rb_opt = rl.RolloutBufferOptions()
ppo_opt = rl.PPOOptions()

policy = rl.Policy(1, 1, policy_opt)
rollout_buffer = rl.RolloutBuffer(1, 1, rb_opt)
ppo = rl.PPO(ppo_opt, policy)

print(policy.parameters())

device = torch.device("cpu")

env = gym.make('CartPole-v0')
obs = env.reset()
done = False

for step in range(10000):
    for _ in range(rb_opt.buffer_size):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(device)
            action_dist, value = policy.forward(obs_tensor)
            value = value.item()
            action = action_dist.sample().cpu().numpy()
            log_prob = action_dist.log_prob(value).item()
        new_obs, reward, new_done, info = env.step(action)
        rollout_buffer.add(0, obs, action, reward, done, value, log_prob)

        obs = new_obs
        done = new_done

    with torch.no_grad():
        obs_tensor = torch.as_tensor(new_obs).to(device)
        _, value = policy.forward(obs_tensor)
        value = value.cpu().numpy()

    rollout_buffer.compute_returns_and_advantage(value, done)
    batches = rollout_buffer.get_samples(128)

    ppo.train(batches)
