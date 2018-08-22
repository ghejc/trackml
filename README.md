# trackml

Python code from the Kaggle TrackML challenge

The implemented algorithm starts from the joint probability distribution *p(x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>,...x<sub>n</sub>|q)* that hits *x<sub>1</sub>,x<sub>2</sub>,x<sub>3</sub>,...x<sub>n</sub>* form a track of a particle with charge *q*. This joint probability density can be factored to *p(x<sub>n</sub>|x<sub>1</sub>,x<sub>2</sub>,..x<sub>n-1</sub>,q)...p(x<sub>2</sub>|x<sub>1</sub>,q)p(x<sub>1</sub>|q)* and each term can be approximated by a Gaussian Mixture Density Network.
