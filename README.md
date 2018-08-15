# trackml

Python code from the Kaggle TrackML challenge

The implemented algorithm starts from the joint probability distribution *p(x1,x2,x3,...xn|q)* that hits *x1,x2,x3,...xn* form a track of a particle with charge *q*. This joint probability density can be factored to *p(xn|x1,x2,..xn-1,q)...p(x2|x1,q)p(x1|q)* and each term can be approximated by a Gaussian Mixture Density Network.
