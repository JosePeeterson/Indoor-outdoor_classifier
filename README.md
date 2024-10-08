This repo. is the result of Machine Learning intership at Continental. see the 'intership.pdf' for more details.

# RL-collision avoidance
Overview
Robot’s travel times to reach their goals depend on a list of possible trajectories from start to goal position, robot’s velocities at each step & collision radius to other robots and obstacle.
To find the optimal policy shared by all robots, Policy gradient based RL called proximal policy optimisation is used to minimize the expectation of the mean travel time of all the robots.
Using the observation each robot independently computes an action sampled from the shared policy.

![image](https://github.com/user-attachments/assets/31f3419b-30a8-48d2-a2b5-2ab31fce0b24)

![image](https://github.com/user-attachments/assets/bd09b630-9fd5-44a5-b115-b9ccf1fc636a)

![image](https://github.com/user-attachments/assets/f68e9124-cd07-4eca-aaa1-3d5d1e5fe610)

## Closed Environment simulation setup for testing

![image](https://github.com/user-attachments/assets/94295829-22c6-493f-88b4-cf15222e89f3)

![image](https://github.com/user-attachments/assets/12491328-865e-4062-bb23-d189967fa514)

![image](https://github.com/user-attachments/assets/617b29f0-f753-4ad6-a49e-037a90b63f58)

## Results
<img width="593" alt="image" src="https://github.com/user-attachments/assets/62fb56a8-4cfc-421b-b725-ce5d7073579c">

# Indoor-Outdoor classifier

<img width="595" alt="image" src="https://github.com/user-attachments/assets/8a264d53-5595-47d9-b3f5-fd09e584ef48">

