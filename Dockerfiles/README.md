# Dockerfiles

| Dockerfile   | Description
|--------------|---------------
| Dockerfile   | Default docker file. Nothing special.
| Dockerfile.cpu_noavx| Building PyTorch from sourcecode for NoGPU and NoAVX.

I have not tested in special hardware environments. If you have any problems, please post them in the issue.
Also, I am not familiar with Docker. If you are familiar with Docker, do it your way.

# Building Docker Image

```
docker build -t nunif Dockerfiles
```

When specifying a docfile (For example `Dockerfiles/Dockerfile.cpu_noavx`)
```
docker build -t nunif -f Dockerfiles/Dockerfile.cpu_noavx Dockerfiles
```

## Running waifu2x.web with larger file limit

```
docker run --gpus all -p 8812:8812 --rm nunif python3 -m waifu2x.web --port 8812 --bind-addr 0.0.0.0 --no-size-limit
```
Open http://localhost:8812/ 

For CPU only mode (use `--gpu -1` option)
```
docker run -p 8812:8812 --rm nunif python3 -m waifu2x.web --port 8812 --bind-addr 0.0.0.0 --no-size-limit --gpu -1
```

## waifu2x.cli command

For CLI commands, it is required to access the input and output directories from the docker container side to the host side.
So, mount the drive on the host side and use it.
```
docker run --gpus all -v path_to_input_dir:/input_dir -v path_to_output_dir:/output_dir --rm nunif python3 -m waifu2x.cli -m scale -i /input_dir -o /output_dir
```
Note that directory paths for `-v` option must be specified as absolute paths.
```
docker run --gpus all -v `pwd`/waifu2x/docs/images:/input_dir -v /tmp/output:/output_dir --rm nunif python3 -m waifu2x.cli -m scale -i /input_dir -o /output_dir
```

See also [waifu2x Command Line Interface](../waifu2x/docs/cli.md) for example commands.
