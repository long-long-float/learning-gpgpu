## Build and run

```console
$ docker build -t gpu .
$ make # it needs nvidia-docker
$ bin/image-process KERNEL lenna.png
```

`KERNEL`: see `image-process.cl`

```console
$ bin/ray-tracing sphere
```

![sphere](result-sphere.png)
