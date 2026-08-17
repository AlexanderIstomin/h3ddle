#!/usr/bin/env python3
"""Export the 8-step flow-match schedule and a stepped trajectory.

The sampler is a dozen lines of arithmetic with four places to be quietly
wrong, and none of them raise:

  - the sigmas are `linspace(1, 1/N, N)` from the pipeline, *not* the
    scheduler's own default spacing;
  - a static shift of 3.0 is then applied. The pipeline also computes a `mu`
    and passes it, which is dead code here because `use_dynamic_shifting` is
    false — copying the mu path instead gives a plausible different schedule;
  - the model is fed `(1000 - t)/1000`, which is 1 - sigma, not sigma;
  - the model output is **negated** before the step.

The trajectory uses a fixed pseudo-model rather than the real one, because
what is under test is the update rule and the schedule, and a real forward
would only make the check slow and the failure ambiguous.

  zimage_sampler_golden.py --out golden.safetensors [--steps 8]
"""

import argparse

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, use_dynamic_shifting=False, shift=3.0
    )
    sigmas = torch.linspace(1.0, 1 / args.steps, args.steps).tolist()
    # mu is what the pipeline passes; it must make no difference here.
    scheduler.set_timesteps(sigmas=sigmas, mu=0.7)
    scheduler.set_begin_index(0)

    timesteps = scheduler.timesteps
    print(f"{args.steps} steps, shift {scheduler.config.shift}")
    print(f"{'i':>2} {'sigma_in':>9} {'sigma':>9} {'t':>9} {'(1000-t)/1000':>14}")
    for i, t in enumerate(timesteps):
        print(f"{i:2d} {sigmas[i]:9.5f} {scheduler.sigmas[i]:9.5f} "
              f"{t:9.4f} {(1000 - t) / 1000:14.5f}")
    print(f"terminal sigma {scheduler.sigmas[-1]:.5f}")

    torch.manual_seed(3)
    latents = torch.randn(1, 16, 8, 8)
    trajectory = [latents.clone()]
    model_inputs = []
    outputs = []
    for i, t in enumerate(timesteps):
        # A fixed, reproducible stand-in for the transformer: the C side gets
        # the same tensors and so exercises the schedule, not the model.
        torch.manual_seed(100 + i)
        model_out = torch.randn_like(latents)
        outputs.append(model_out.clone())
        model_inputs.append(torch.tensor([(1000 - t) / 1000]))
        latents = scheduler.step(-model_out, t, latents, return_dict=False)[0]
        trajectory.append(latents.clone())
        print(f"  step {i}: |latent| {latents.abs().mean():.5f}")

    tensors = {
        "sigmas": scheduler.sigmas.clone(),
        "timesteps": timesteps.clone(),
        "model_input_t": torch.cat(model_inputs),
        "initial": trajectory[0].squeeze(0),
        "final": trajectory[-1].squeeze(0),
        "outputs": torch.stack(outputs).squeeze(1),
        "trajectory": torch.stack(trajectory).squeeze(1),
    }
    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
