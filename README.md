# REPORT:
https://docs.google.com/document/d/1aMJw1OJJ9xC9KH4iz8jccAXoKYUwuLNT66qoMKNH0cY/edit?usp=sharing 

# INSTRUCTION TO RUN INFERENCE WITH DOCKER


## Download test B dataset
Download dataset at https://drive.google.com/file/d/1Q7kAqFITGntZAn-HuCh8vQpHTpDkPSAH/view?usp=sharing and unzip

## Set environment variables for path to dataset

```bash
export DATASET_DIR=<absolute_path_to_dataset_dir>
```

For example:

```bash
export DATASET_DIR=/media/ocr/external/duyla/RESEARCH/TEXTRECOGNITION/WORDART/ICDAR24-WordArt_testB/
```

## Pull docker image

```bash
docker pull duybktiengiang1603/wordart:v1.3
```

## Run docker container and mount source code and dataset

```bash
docker run --rm --gpus all -v $DATASET_DIR:/dataset/ --name wordart -dt duybktiengiang1603/wordart:v1.3
```

```bash
docker exec -it wordart /bin/bash
```

## Run infererence


```bash
cd workdir
```

```bash
source activate duyla_parseq
```

```bash
bash script/infer.sh
```

```bash
bash script/ensemble.sh
```

Final result is located in `submission/test_B/recheck/comb1/answer.txt`
