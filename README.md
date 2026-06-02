This is a first experimental repository, built to be a testing ground for initial ideas and experiments (so, despite the effort, possibly a little messy and not so clean in comments). The code also might be a bit messy, but it serves as a sandbox for learning and prototyping. Every script is heavily commented (if you still find some Italian I'm sorry. Feel free to contact me) to clearly highlight its purpose. In the following the instructions to use the more stable and mature scripts you can use as an experimental pipeline:


-instance_generator.py: to generate instances we want to embed. For atom neutral QPUs with global ray like Jade, use the generate_verisimilar_UDG_qubo method with desired parameters.

-inspect_qubo.py: to inspect visually (in console) and mathematically the QUBO matrix we are interested in.

INSIDE JADE:

-bench.py: is the "core", load the instance we want to embed and send to resolve it to AnalogQPU. In detail{

    1. instance loading
    2. parametrize the GA as you like with the proposed fitness function and hyperparameter of your choice (I recommend improved_topological_mae_fitness_func as I obtained good results on 10x10 UDG instances).
    3. send the job on the remote Julich HPC (Jade is not ready, so I simulate with AnalogQPU but with Jade's constraints).
    4. log everything on .csv.

}

-retrieve.py: final step of the experimental pipeline for Jade. Insert the Job_id, the instance file path, execute and an image with probabilities will be shown. If optimal solutions have been found, the bar charts are green. It logs in the same .csv inserted in bench.py and copied in this script.

The other scripts in jade are very experimental and not so mature. Their aim was to tests some possible variants, feel free to explore.

I uploaded my_QUBO_instances to show what instances I worked with (in this moment I'm focused on UDG ones), but I can also share hamburgh dataset.


INSIDE D-Wave:

work in progress to build an experimental pipeline, but is already possible to send the interested tasks and obtain some logs.


Feel free to contact me for any detail, observation, critic or idea. I really appreciate any sort of contribution/constructive exchange of ideas.



