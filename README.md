# Warp Sim Pole Balancing learner

I wanted to write the classic pole balancing problem using warp as the simulation environment.
You can read about warp here: https://github.com/NVIDIA/warp 

To get started install the following...
```bash
pip install -r requirements.txt

# Install pytorch for you environment
https://pytorch.org/get-started/locally/
```


**The following examples show how I set up the environment.**


- **Example 0**
    
    Basic pole swinging around a box with a set inital joint quaternion (one degree of freedom)

- **Example 1**

    Modified problem so that pole drops at random angle (3 degrees of freedom)
    
    <img width="500" alt="Figure_0" src="https://github.com/user-attachments/assets/c8c8471a-6094-42be-902a-cc935135366f" />

- **Example 2**

    Created a warp kernel to move the body and pass the positions of a circle.
    You can see the outward force pushing on the pole as it revolves.

    [revolve.webm](https://github.com/user-attachments/assets/3c30f29f-1f0c-49dd-91c6-de6fabdc8a74)

- **Example 3**

    Capturing the pole state as a vector. Ploting pole position, velocity and angle.

    <img width="600" alt="Figure_1" src="https://github.com/user-attachments/assets/4bb8ca94-f9e3-4161-9991-3202a1a954e2" />


- **Example 4**

    Training a nueral network to balance the pole based on the simulation (one degree of freedom).
    The network is able to act on the velocity of the body controlling the cart. I use an actor critic design.

    First video is the network untrained second is after training.
    You can try to balance the pole your-self by running the simulation with the `--manual-control"` flag and `k` and `l` keys.

    [untrained.webm](https://github.com/user-attachments/assets/495b33e6-e1ef-4e6f-9bac-f2f4401c0c5a)

    [trained.webm](https://github.com/user-attachments/assets/5b6c0918-146b-4a10-93de-6ec683dcbc4e)


