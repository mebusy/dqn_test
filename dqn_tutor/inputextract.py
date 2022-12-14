import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

# The code below are utilities for extracting and processing rendered images from the environment.
# It uses the torchvision package, which makes it easy to compose image transforms.
# Once you run the cell it will display an example patch that it extracted.

resize = T.Compose(
    [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
)


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2, cart_location + view_width // 2
        )
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


if __name__ == "__main__":
    import gym
    import matplotlib
    import matplotlib.pyplot as plt

    env_test = gym.make("CartPole-v0", render_mode="rgb_array").unwrapped
    env_test.reset()
    plt.figure()
    plt.imshow(
        get_screen(env_test).cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation="none",
    )
    plt.title("Example extracted screen")
    plt.show()
