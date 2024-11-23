# Collection of LLM Benchmarks for Jetson Orin Nano

This is a collection of LLM Benchmarks to run on NVIDIA Jetson Orin Nano on Jetpack V6.1.

Currently I'm experimenting using OPT and Llama model. More model and better code will be added.

a.k.a Work In Progress

## Installation 
### Make new venv
First, make new venv to work with.

```bash
python3 -m venv env
```

### Install Dependencies
Install pytorch for Jetson device, for more detailed steps, go to [NVIDA Docs Hub](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

For this project, I'll be using pytorch build from [devpi(?)](http://jetson.webredirect.org/jp6/cu126).

Install dependencies on Jetson device,

```bash
pip install -r requirements_edge.txt
```
or for PC for debugging

```bash
pip install -r requirements.txt
```

### Install Jetson-stats
For performance logging I'll be using [Jetson-stats](https://github.com/rbonghi/jetson_stats/tree/master)

Install it using ***Superuser***

```bash
sudo pip3 install -U jetson-stats
```
```bash
python3 jtop_logger.py --file 
```

## Code

There are currently 2 code, Latency testing and text output testing.

### Latency Test
For latency it can run by running below code

```bash
python3 opt_eval_latency.py --model facebook/opt-350m --seed 0 --token_size 200
```

* ``--model`` OPT Model used for inference
* ``--seed`` Seed used forinference
* ``--token_size`` The size of generated token used for inference

```bash
python3 llama_eval_latency.py --model baffo32/decapoda-research-llama-7B-hf --seed 0 --token_size 200
```

* ``--model`` Llama Model used for inference
    * It is best to use sharded model if you want to run it on Edge
* ``--seed`` Seed used forinference
* ``--token_size`` The size of generated token used for inference

For the text prompt used, you can change it by changing the ``prompt`` variable on the python files.


### Text Output
This code is just a test if the code it self can output a text.

```bash
python3 text_output_opt.py
```

or

```bash
python3 text_output_llama.py
```
Still WIP