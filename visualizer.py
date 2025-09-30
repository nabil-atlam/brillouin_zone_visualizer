#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np 
from   numpy import pi, array

# Plotting library for 3D visualization
import plotly.graph_objects as go
import plotly.io as pio 
# Computational geometry  
from scipy.spatial import Voronoi, ConvexHull 




# In[113]:


# Usage 
# ========================================================================================================================
# This script first computes a set of equally spaced parallel k-planes 
# in the momentum space and writes a bunch of wannier tools input files to compute the iso-energy surfaces on each k-plane. 
# ========================================================================================================================


# Input parameters 
# ======================================================================================================================== 

# Real Space Basis as a Numpy array 
real_basis = array([[-1.9287500381500000,    1.9287500381499998,    4.7979998588499999],
            [1.9287500381500000,   -1.9287500381500005,    4.7979998588499999],
            [1.9287500381500000,    1.9287500381500005,   -4.7979998588499999]])


# In[125]:


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def hex_to_rgba(hex_color, alpha = 1.0):
    r, g, b = hex_to_rgb(hex_color)
    return f'rgba({r}, {g}, {b}, {alpha})'

def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:  # Not a Jupyter kernel
            return False
    except Exception:
        return False
    return True


def get_the_2_simplices_sharing_an_edge(simplices, edge):
    """Find all 2-simplices (triangles) that share a given edge. 
    """
    matching_simplices = []
    for simplex in simplices:
        if edge[0] in simplex and edge[1] in simplex:
            matching_simplices.append(simplex)
    return matching_simplices


def get_edges_convexhull(hull):
    """Extract edges from the ConvexHull object.
    """
    edges = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % len(simplex)])))
            edges.add(edge)
    return list(edges)
     
def compute_normal_simplex(simplex3_cartesian):
    p1 = simplex3_cartesian[0] 
    p2 = simplex3_cartesian[1] 
    p3 = simplex3_cartesian[2] 
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return normal


# In[184]:


class kplane_plotter:
    def __init__(self, real_basis, miller_indices, tol_fbz_plot = 0.01):
        '''
        real_basis: 3x3 matrix, each row is a lattice vector in real space 
        miller_indices: Array  of 3 integers, the Miller indices of the planes  
        '''
        if real_basis.shape != (3,3):
            raise ValueError("real_basis must be a 3x3 matrix")
        if len(miller_indices) != 3:
            raise ValueError("miller_indices must be an array of 3 numbers")
        
        self.real_basis     = real_basis 
        self.miller_indices = miller_indices 

        # Compute the reciprocal basis vectors 
        self.recip_basis  = self.compute_recip_basis() 
        self.G1           = self.recip_basis[0] 
        self.G2           = self.recip_basis[1]
        self.G3           = self.recip_basis[2]

        _normal_vec = self.miller_indices[0] * self.G1 + self.miller_indices[1] * self.G2 + self.miller_indices[2] * self.G3 
        self.normal_vec = _normal_vec / np.linalg.norm(_normal_vec) 

        # Next, we want two vectors parallel to the plane. There is a simple algorithm for doing that 
        self.kplane_vec_1 = self.compute_perp_vec(self.normal_vec) 
        # The third one is the cross product of the first two 
        self.kplane_vec_2 = np.cross(self.normal_vec, self.kplane_vec_1)

        self.tol_fbz_plot = tol_fbz_plot 

    ##########################################################################################
    # Helper functions
    ##########################################################################################
    def compute_recip_basis(self):
        '''
        Compute the reciprocal basis from real basis using the standard formula  
        '''
        return 2.0 * pi * np.linalg.inv(self.real_basis).T 

    
    def compute_perp_vec(self, vec):
        '''
        Given a vector, obtain a vector perpendicular to it 
        '''
        if vec[0] == 0 and vec[1] == 0:
            vec = array([1.0, 0.0, 0.0])
        else:
            vec = array([-vec[1], vec[0], 0.0])

        return vec / np.linalg.norm(vec) 
    

    def calculate_grid_recip(self, cutoffs):
        '''
        Given a set of reciprocal lattice vectors and cutoffs, compute a grid in the reciprocal space 
        Gs: 3x3 matrix, each row is a reciprocal lattice vector
        cutoffs: array of 3 integers, the cutoff in each direction 
        '''
        grid       = np.mgrid[-cutoffs[0]:cutoffs[0] + 1, -cutoffs[1] : cutoffs[1] + 1, -cutoffs[2] : cutoffs[2] + 1]
        xs, ys, zs = np.tensordot(self.recip_basis, grid, axes = (0, 0))
        ps         = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]          # Column stack, so the three columns are X, Y, Z coordinates 
        

        
        return ps
     
    def calc_fbz_vertices(self, cutoffs):
        ps = self.calculate_grid_recip(cutoffs)
        vor3 = Voronoi(ps) 

        # Compute the vertices 
        # Note: There are multiple Voronoi regions, we need to find the one that contains the origin 
        # First step, find the index of the point closest to the origin. 
        origin_idx = np.argmin(np.linalg.norm(ps, axis = 1))
        print(f"--- Found the index of the point closest to the origin: {origin_idx}")

        # Next, find the region corresponding to this point 
        id_region = vor3.point_region[origin_idx]
        print(f"--- Found region containing  the origin point: {id_region}")

        fbz_verts_idx = vor3.regions[id_region] 
        print(f"--- The vertices of this region are: {[ind for ind in fbz_verts_idx]}")

        fbz_verts_set = set(fbz_verts_idx)
        verts_bz      = vor3.vertices[fbz_verts_idx] 
        for i, v in enumerate(verts_bz):
            print(f"Vertex {i}: {v}")

        hull          = ConvexHull(verts_bz) 
        simplices     = hull.simplices

        
        edges = get_edges_convexhull(hull) 
        
        
        edges_cartesian = [] 
        for e in edges:
            p1 = verts_bz[e[0]]
            p2 = verts_bz[e[1]]
            edges_cartesian.append([p1, p2]) 

        edges_final      = [] 
        edges_final_cart = [] 

        # Now, we have found the edges. This leads to a triangulation of the boundary surface of the FBZ 
        for e in edges: 
            simplices_ = get_the_2_simplices_sharing_an_edge(simplices, e) 
            if len(simplices_) != 2:
                raise ValueError("Error occured in <get_the_2_simplices_sharing_an_edge>")
            s1   = simplices_[0]
            s2   = simplices_[1]
            n1   = compute_normal_simplex(hull.points[s1])
            n2   = compute_normal_simplex(hull.points[s2])
            n1n2 = np.dot(n1, n2) 
            
            if 1.0 - np.abs(n1n2) > self.tol_fbz_plot:
                edges_final.append(e)
        print(f"--- Found {len(edges_final)} edges of the FBZ that do not share coplanar simplices")

        for e in edges_final:
            p1 = verts_bz[e[0]]
            p2 = verts_bz[e[1]]
            edges_final_cart.append([p1, p2]) 
                
        

                
        print(f"--- Found {len(edges_cartesian)} edges of the FBZ")
        return edges_final_cart 

    
    
    
    # Now, I need to write the plotting code. 
    # Requirements: 
    # ======================================================================================================== 
    # 1. Given the two plane vectors and a sequence of origin points, compute the k-planes and plot them 
    # 2. Plots the FBZ as well 
    # 3. The plot should be clear and easily understandable 
    # -======================================================================================================= 
    def plot(self):

        # basic things about the figure dimensions 
        f_width = 1200 
        f_height = f_width

        # Will use plotly for 3D plotting 
        f = go.Figure() 
        

        # Arrays to define the Brillouin Zone basis vectors. Will plot as three arrows with cones at the tip 
        npts_shaft = 100 
        G_ = [self.G1, self.G2, self.G3] 

        Gs_x = [G_[i][0] for i in range(3)] 
        Gs_y = [G_[i][1] for i in range(3)]
        Gs_z = [G_[i][2] for i in range(3)]

        xs = []; ys = []; zs = []

        for i in range(3):
            xs.append(np.linspace(0, G_[i][0], npts_shaft))
            ys.append(np.linspace(0, G_[i][1], npts_shaft))
            zs.append(np.linspace(0, G_[i][2], npts_shaft))
             
        rgb_bzshafts = hex_to_rgba('#192a56', 0.8)

        # Plotting code 
        for i in range(3):
            f.add_trace(go.Scatter3d(
                x = xs[i], y = ys[i], z = zs[i],
                mode = 'lines',
                line = dict(color = rgb_bzshafts, width = 30),
                name = 'BZBasis_Shaft')
                )
            
        # Plot the cones at the tips of the arrows 
        f.add_trace(
                go.Cone(
                x = Gs_x, y = Gs_y, z = Gs_z,
                u = Gs_x, v = Gs_y, w = Gs_z,
                colorscale = [[0, rgb_bzshafts], [1, rgb_bzshafts]],
                showscale  = False,
                sizemode   = "scaled",
                sizeref    = 0.1,  
                anchor     = "tail")
                )
        
        edges = self.calc_fbz_vertices(cutoffs = [1,1,1])
        for p in edges:
            p1 = p[0] 
            p2 = p[1]
            ex = [p1[0], p2[0]]
            ey = [p1[1], p2[1]]
            ez = [p1[2], p2[2]]
            f.add_trace(go.Scatter3d(
                x = ex, y = ey, z = ez,
                mode = 'lines',
                line = dict(color = "#2f3640", width = 10),
                name = 'BZ edges')
                )
        
        

        # Plot the Brillouin zone edges 
        # This is nontrivial because we need to calculate the coordinates of the vertices at the boundaries of the BZ 


        f.update_layout(
            showlegend = False, 
            width = f_width,
            height= f_height,
            scene = dict(
            xaxis = dict(visible = False),
            yaxis = dict(visible = False),
            zaxis = dict(visible = False))
            )
        if in_notebook():
            pio.renderers.default = "notebook"   #
        else:
            pio.renderers.default = "browser"    #
        f.show()
    
        
        
        
    


# In[ ]:


def main():
    plotter = kplane_plotter(real_basis, [1,1,-1]) 
    plotter.plot()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




