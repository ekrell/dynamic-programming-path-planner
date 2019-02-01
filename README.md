# dynamic-programming-path-planner
Implementation of dynamic programming mobile robot path planner from paper by Kwan S. Kwok and Brian J. Driessen
 
        Kwok, K.S., and Driessen, B.J.. Path planning for complex terrain navigation via dynamic programming. United States: N. p., 1998. Web. 

Note that the map, start location, and goal location are currently hard-coded. I will soon make it into a more useful tool that accepts command line arguments.

I was interested in using dynamic programming (DP) for path planning because it can be done such that you have an optimal path to goal from every cell in the space. This is unlike the typical path planner that gives a single solution path from a single start location. However, the use of DP within is just for a single path. 
