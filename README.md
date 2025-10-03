# Warp Sim Pole Balancing learner

I wanted to write the classic pole balancing problem using warp as the simulation environment.
You can read about warp here: https://github.com/NVIDIA/warp 

To get started install the following...
```bash
pip install -r requirements.txt

# Install pytorch for you environment
https://pytorch.org/get-started/locally/
```


The following examples show how I set up the environment.


- Example 0
    
    Basic pole swinging around a box with a set inital joint quaternion (one degree of freedom)

- Example 1

    Modified problem so that pole drops at random angle (3 degrees of freedom)

- Example 2

    Created a warp kernel to move the body and pass the positions of a circle.
    You can see the outward force pushing on the pole as it revolves.

- Example 3

    Capturing the pole state as a vector. Ploting pole position, velocity and angle.

- Example 4

    Training a nueral network to balance the pole based on the simulation (one degree of freedom).
    The network is able to act on the velocity of the body controlling the cart. I use an actor critic design.


