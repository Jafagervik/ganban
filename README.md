# Pytorch project template 

![alt text](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fadityassrana.github.io%2Fblog%2Fimages%2Fcopied_from_nb%2Fimages%2Fmeme2.png&f=1&nofb=1&ipt=c660ae09b7a4e4e30eb266059751d9b9a52fbc0d95a3d04a63435fa7ae7d8965&ipo=images)



## Motivation

I don't want to write the same boring dataloaders all over again
I just want to be able to transform cool ideas in to code. 

Will probably use this in the course "Pattern Recognition" at Waseda 

## Structure

### model.py

Contains *ONLY* your model


### train.py

Entry point. here we just mix everything and run it all 


### detect.py

TODO: must be implemented before anything else 

### config.py

Contains all hyperparameters
And yes, we're using a python file and not yaml 

### data 

Folder to store your train, test and val images

### utils/helpers.py

Annoying code you need to get up and running

### utils/visuals.py

Don't ask me, I am not a data scientist or analyst
Some cool plotting goes into here 

### datasetup.py

Set up and install your data inside here

### eval.py

Run your own data here to only test
