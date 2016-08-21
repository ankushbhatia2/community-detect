# community-detect
Community detection using attribute and structural similarities.


Installation:
python setup.py install

Dependencies:
NetworkX
Matplotlib

Usage:
Import:
from community_detect import Community

Initialize:
com = Community(alpha_weight = 0.5) #You can add your own value for Alpha

Functions:
Main method: get_communities(Graph, #Your Graph
                             Vertices, #List of Vertices
                             Similarity Matrix, #Similarity matrix for attribute similarities (It should be a N X N matrix where N is the number of vertices
                             Similarity Matrix Type #Types : cosine, euclidean etc
                             )
            Returns a dictionary with each key containing all the nodes in that community
To View Communities : view_communities(Communities, #Output of above function
                                       Graph, #Your Graph
                                       Vertices, #List of Vertices
                                       Similarity Matrix, #Similarity matrix for attribute similarities (It should be a N X N matrix where N is the number of vertices
                                       Similarity Matrix Type #Types : cosine, euclidean etc
                                       )
