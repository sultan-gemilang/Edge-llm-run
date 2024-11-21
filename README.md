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

### Install Pytorch
Install pytorch for Jetson device, for more detailed steps, go to [NVIDA Docs Hub](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

```bash
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

```bash
python3 -m pip install --upgrade pip;
python3 -m pip install numpy==’1.26.1’;
python3 -m pip install --no-cache $TORCH_INSTALL
```

### Install other dependencies
The other dependencies for this roject can be install using requirements.txt

```bash
pip install -r requirements.txt
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

 For the text prompt used, you can change it by changing the ``prompt`` on the python files.


### Text Output
This code is just a test if the code it self can output a text.

```bash
python3 text_output_opt.py
```