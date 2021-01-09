//
// Created by lasagnaphil on 21. 1. 8..
//

#ifndef FASTRL_CARTPOLE_ENV_H
#define FASTRL_CARTPOLE_ENV_H

#include <random>
#include <tuple>
#include <raylib.h>

inline float angle_normalize(float x) {
    return std::fmod(x + (float)M_PI, 2*(float)M_PI) - (float)M_PI;
}

#define SQUARE(x) ((x)*(x))

// A continuous version of the famous CartPole environment.
struct CartpoleEnv {
    constexpr static int obs_dim = 4;
    constexpr static int act_dim = 1;

    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float total_mass = 1.0f + 0.1f;
    float length = 0.5f;
    float polemass_length = 0.1f * 0.5f;
    float force_mag = 10.0f;
    float tau = 0.02f;
    float theta_threshold_radians = 12.f * 2.f * M_PI / 360;
    float x_threshold = 2.4f;

    int steps_beyond_done = -1;

    int screen_width = 800;
    int screen_height = 600;

    std::array<float, obs_dim> state;
    std::default_random_engine random_engine;

    bool continuous_action_space;

    CartpoleEnv(bool continuous_action_space = false) : continuous_action_space(continuous_action_space) {
        seed();
        reset();
    }

    void seed(unsigned long seed = 0) {
        random_engine.seed(seed);
    }

    std::tuple<std::array<float, 4>, float, bool> step(float action) {
        auto [x, x_dot, theta, theta_dot] = state;
        float force;
        if (continuous_action_space) {
            force = std::min(std::max(force_mag * action, -force_mag), force_mag);
        }
        else {
            force = action > 0.5f? force_mag : -force_mag;
        }
        float costheta = std::cos(theta);
        float sintheta = std::sin(theta);

        float temp = (force + polemass_length * SQUARE(theta_dot) * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) /
                (length * (4.0f / 3.0f - masspole * SQUARE(costheta) / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;

        state = {x, x_dot, theta, theta_dot};

        bool done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
        float reward;
        if (!done) {
            reward = 1.0f;
        }
        else if (steps_beyond_done == -1) {
            steps_beyond_done = 0;
            reward = 1.0f;
        }
        else {
            if (steps_beyond_done == 0) {
                fprintf(stderr, "You are calling 'step()' even though this "
                                "environment has already returned done = True. You "
                                "should always call 'reset()' once you receive 'done = "
                                "True' -- any further steps are undefined behavior.\n");
            }
            steps_beyond_done++;
            reward = 0.0f;
        }
        return {state, reward, done};
    }

    std::array<float, 4> reset() {
        for (int i = 0; i < 4; i++) {
            state[i] = std::uniform_real_distribution<float>(-0.05, 0.05)(random_engine);
        }
        steps_beyond_done = -1;
        return state;
    }

    void render() {
        auto [x, x_dot, theta, theta_dot] = state;
        float scale = 400;
        float pole_width = 20;
        float pole_height = scale * length;
        float pos_x = screen_width*0.5f - pole_width*0.5f + scale*x;
        float pos_y = screen_height*0.5f;
        Rectangle pole_rect {pos_x, pos_y, pole_width, pole_height};
        Vector2 pole_origin {pole_width*0.5f, 0};
        DrawRectanglePro(pole_rect, pole_origin, RAD2DEG * (theta + M_PI), RED);
        DrawCircle(pos_x, pos_y, pole_width, GRAY);
    }

};


#endif //FASTRL_CARTPOLE_ENV_H
