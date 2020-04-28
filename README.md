# CS4278-5478-Project-Materials

This repository contains the evaluation materials for the CS4278/CS5478 project, autonomous lane following in gym-duckietown.

## Maps
We will evaluate your policy on 5 maps. Please find them [here](./maps/), then copy them into the map folder of your gym-duckietown environment, e.g.,
```
cp maps/* /path/to/your-gym-duckietown-repo/gym-duckietown/maps/
```

## Evaluation  

When you initialize the duckietown environment, you should add three additional arguments:
- `--map-name`: the name of the map
- `--max-steps`: the maximum run step. The default value is 2000. Please do not change this.
- `--seed`: random seed of the environment. 

Similar to Assignment 3, you should generate the control files for submission. Each map is associated with 10 random seed. Please test your policy with random seed **from 1 to 10**. You should generate control files for each random seed and map. In particular, there are several [invalid seeds](./invalid_seeds.json) for each map. Please skip them and test the rest.

A sample file for the environment can be found [here](./example.py). We also include a [sample control file]('./../map5_seed11.txt). We illustrate how to add arguments, dump your controls into a file in the example. To try our simple policy, please do:
```
python example.py --map-name map5 --seed 11
```

We will compute the accumulated reward for each test case, and grade your project based on the average reward achieved. 

## Submission
Save your controls following the naming convention: map{map_number}_seed{seed_number}.txt. For example:
- map1_seed5.txt
- map2_seed1.txt
- map5_seed11.txt (this is our sample file. You don't need to consider seed 11)

You also should include an up-to-2-page report to briefly explain your method. 

Please organize your submission folder with the following structure:
```
student_id.zip
|-- report.pdf
|-- code
|-- control_files
    |-- map*_seed*.txt
    |-- ...
```
