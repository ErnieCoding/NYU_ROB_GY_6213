##use carla environment for testing the joystick:

import pygame
import time

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Connected:", joystick.get_name())

while True:
    pygame.event.pump()

    # Try these indices (may vary slightly)
    r2 = joystick.get_axis(5)   # Right trigger
    l2 = joystick.get_axis(4)   # Left trigger
    steer = joystick.get_axis(0)  # Left stick X

    # Normalize trigger: [-1,1] → [0,1]
    r2_val = (r2 + 1) / 2
    l2_val = (l2 + 1) / 2

    # Combine into forward/backward speed
    speed = r2_val - l2_val

    print(f"Speed: {speed:.2f} | Steering: {steer:.2f}")

    time.sleep(0.1)