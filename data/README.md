After downloading the [th-1](https://crcns.org/data-sets/thalamus/th-1) dataset, selecting `Mouse28` and session `140313`, run the preprocessing script 

```
python3 preprocess_dataset.py --datadir /scratches/sagarmatha_2/ktj21/data/crcns/ --savedir /scratches/ramanujan_2/dl543/preprocessed/th1/
```


We load the dataset for one session from the original files downloaded from [CRCNR.org](https://crcns.org/), which is then preprocessed and put into a pickle file in the ```../data/datasets``` directory. We look at a dataset from Mouse 28:


    Postsubiculum:
    Electrode groups 1-7.
    6 shank probe, Neuronexus Buz64sp design. Shank #1 is the most medial, #6 the most lateral, group #7 made
     of the four sites located above 4th shank (see design).
    Approx depth from surface: 1.28mm

    Anterior thalamus:
    Electrode groups 8-11.
    4 shank probe, Neuronexus Buz32 design. Shank #8 is the most lateral, #11 the most medial.
    Approx depth from surface: 2.56mm


Running `gen_synthetic_data.py` overwrites the previous files.


```
python3 select_th1_units.py --savedir /scratches/ramanujan_2/dl543/preprocessed/th1/ --datadir /scratches/ramanujan_2/dl543/preprocessed/th1/
```