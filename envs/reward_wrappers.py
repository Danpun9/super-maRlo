import gymnasium as gym


class CustomMarioRewardWrapper(gym.Wrapper):
    """
    Precision Reward Wrapper for Super Mario Bros:
    - Protects early/mid game trajectories (0 < x < 2800) with standard reliable rewards.
    - Eliminates staircase freezing (x >= 2800) with severe anti-stagnation penalties.
    - Heavily incentivizes jump actions (Action 2: right+A, Action 4: right+A+B) on the stairs.
    - Large milestone bonuses for clearing the pyramid peak and descending into the flagpole.
    """
    def __init__(self, env, max_stagnant_steps=100):
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.current_x = 40
        self.max_x = 40
        self.stagnant_steps = 0
        self.prev_score = 0
        self.prev_status = "small"
        self.prev_lives = 2
        # Endgame milestone flags
        self.passed_3000 = False
        self.passed_top_3050 = False
        self.passed_descend_3100 = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_x = 40
        self.max_x = 40
        self.stagnant_steps = 0
        self.prev_score = 0
        self.prev_status = "small"
        self.prev_lives = 2
        self.passed_3000 = False
        self.passed_top_3050 = False
        self.passed_descend_3100 = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_pos = info.get("x_pos", self.current_x)
        score = info.get("score", self.prev_score)
        flag_get = info.get("flag_get", False)
        lives = info.get("life", self.prev_lives)

        custom_reward = 0.0

        # 1. Forward movement reward
        delta_x = x_pos - self.current_x
        if delta_x > 0:
            mult = 3.0 if x_pos >= 2800 else 1.0
            custom_reward += delta_x * mult
        elif delta_x < 0:
            custom_reward += delta_x * 1.5

        # 2. Update max_x & stagnation tracking
        if x_pos > self.max_x:
            self.max_x = x_pos
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        # 3. Endgame Staircase Logic (x >= 2800)
        if x_pos >= 2800:
            # Reward jump inputs on stairs
            if action in [2, 4]:
                custom_reward += 2.0
            elif action == 1:
                custom_reward -= 1.0  # Punish pushing against the block without jumping

            # Severe anti-freezing penalty on stairs (never allow idling!)
            if self.stagnant_steps > 15:
                custom_reward -= 3.0
            if self.stagnant_steps > 45:  # Truncate after ~1.5s of no progress on stairs
                truncated = True
                custom_reward -= 30.0

            # Progressive milestones
            if x_pos > 3000 and not self.passed_3000:
                self.passed_3000 = True
                custom_reward += 40.0

            if x_pos > 3050 and not self.passed_top_3050:
                self.passed_top_3050 = True
                custom_reward += 80.0  # Big bonus for scaling the top

            if x_pos > 3100 and not self.passed_descend_3100:
                self.passed_descend_3100 = True
                custom_reward += 150.0  # Massive dive incentive toward flagpole
        else:
            # Normal midgame stagnation handling
            if self.stagnant_steps > 30:
                custom_reward -= 0.5
            if self.stagnant_steps > self.max_stagnant_steps:
                truncated = True
                custom_reward -= 10.0

        # 4. Score bonus (enemies, coins)
        score_diff = score - self.prev_score
        if score_diff > 0:
            custom_reward += score_diff * 0.05

        # 5. Death penalty
        if terminated and not flag_get:
            custom_reward -= 50.0

        # 6. Flag clear jackpot!
        if flag_get:
            custom_reward += 500.0
            terminated = True

        # Update state trackers
        self.current_x = x_pos
        self.prev_score = score
        self.prev_lives = lives

        # Scale reward
        scaled_reward = custom_reward / 10.0

        return obs, scaled_reward, terminated, truncated, info
