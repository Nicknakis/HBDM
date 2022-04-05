

import torch
from fractal_kmeans_cond import Euclidean_Kmeans
#from fractal_kmeansSQ import Euclidean_Kmeans

from torch_sparse import spspmm
from copy import deepcopy
class Tree_kmeans_recursion():
    def __init__(self,minimum_points,init_layer_split):
        """
        Kmeans-Euclidean Distance minimization: Pytorch CUDA version
        k_centers: number of the starting centroids
        Data:Data array (if already in CUDA no need for futher transform)
        N: Dataset size
        Dim: Dataset dimensionality
        n_iter: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        full_cuda_load: send the whole matrix into CUDA for faster implementations (memory requirements)
        
        """
        self.minimum_points=minimum_points
        self.init_layer_split=init_layer_split
        self.thetas=[]
        self.max_layers=3*init_layer_split
        self.cond_control=0.001
        self.general_mask=[]
        self.general_cl_id=[]
        
    def kmeans_tree_z_initialization(self,depth,initial_cntrs):
        self.leaf_centroids=[]


        #first iteration
        #initialize kmeans with number of clusters equal to logN
        flag_uneven_split=False
        model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= self.spectral_data.shape,init_cent=initial_cntrs,device=self.device)
        sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(self.spectral_data)
        full_prev_cl=cl_idx
        #flags shows if all clusters will be partioned later on
        split_flag=1
        global_cl=torch.zeros(self.spectral_data.shape[0]).long()
        initial_mask=torch.arange(self.spectral_data.shape[0])
        init_id=0
        sum_leafs=0
        #self.thetas.append(theta_approx)
        for i in range(depth):
            if i==self.max_layers:
                self.cond_control=10*self.cond_control
                print(self.cond_control)
            if i>29:
                print(i)
        #datapoints in each of the corresponding clusters
            #Compute points in each cluster
            assigned_points= torch.zeros(sparse_mask.shape[0])
            assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            #Splitting criterion decides if a cluster has enough points to be binary splitted
            self.splitting_criterion=(assigned_points>self.minimum_points)
            #print(assigned_points)
            #datapoints required for further splitting
            self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())
            self.mask_split=self.mask_split.squeeze(-1).bool()
            #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
            split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])

            #if not all of the clusters have enough points to be partioned
            # save them as leaf nodes and create the mask for the analytical calculations
            if not split_flag:
               
                #erion shows the leaf nodes
                erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
                #if we have leaf nodes in this tree level give them global ids
                if erion.sum()>0:
                    #keep the leaf nodes for visualization
                    with torch.no_grad():
                        self.leaf_centroids.append(model.centroids[erion])
                    #sum_leafs makes sure no duplicates exist in the cluster id
                    sum_leafs=sum_leafs+erion.sum()
                    
                    #global unique cluster ids to be used
                    clu_ids=torch.arange(init_id,sum_leafs)
                    #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
                    cl_vec=torch.zeros(sparse_mask.shape[0]).long()
                    cl_vec[erion]=clu_ids
                    #initial starting point for next level of ids, so no duplicates exist
                    init_id=sum_leafs
                    # print(global_cl.unique(return_counts=True))
                    # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
                    mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
                    # gl_idx takes the global node ids for the mask
                    gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
                    gl_idx=gl_idx.squeeze(-1).bool()
                    # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
                    gl_idx2=gl_idx
                    if gl_idx.shape[0]!=global_cl.shape[0]:
                        #in the case of uneven split the initial mask keeps track of the global node ids
                        gl_idx=initial_mask[gl_idx].long()

                    # give the proper cl ids to the proper nodes
                    global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
                    self.K_leafs=sum_leafs
           
 
            centers=model.centroids
            if self.splitting_criterion.sum()==0:
                break
            
            #local clusters to be splitted
            splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

            if not split_flag:
                flag_uneven_split=True
                # rename ids so they start from 0 to total splitted
                splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
                # create sparse locations of K_old x K_new
                index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
                # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
                self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask=initial_mask[self.ind[0,:].long()]
                self.mask_split=initial_mask
                cl_idx=self.ind[1,:].long()
            if flag_uneven_split:
                #translate the different size of splits to the total N of the network
                self.mask_split=initial_mask
            model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.spectral_data[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,device=self.device)
            sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(self.spectral_data[self.mask_split])
            full_prev_cl=cl_idx
        return global_cl,torch.cat(self.leaf_centroids)
        
    
    
    def kmeans_tree_scaling(self,depth,initial_cntrs):
        self.leaf_centroids=[]
        self.general_mask=[]
        self.general_cl_id=[]
        self.general_centroids_sub=[]
        
        self.missing_center_i=[]
        self.missing_center_j=[]
        self.removed_bias_i=[]
        self.removed_bias_j=[]
        self.total_K=int(self.init_layer_split)


        #first iteration
        #initialize kmeans with number of clusters equal to logN
        flag_uneven_split=False
        model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= self.latent_z.shape,init_cent=initial_cntrs,device=self.device)
        sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.scaling_factor.detach()*self.latent_z.detach()),self.scaling_factor*self.latent_z.detach())
        self.general_mask.append(torch.arange(self.input_size))       

        self.general_cl_id.append(cl_idx)
        self.centroids_layer1=model.centroids
        full_prev_cl=cl_idx
        first_centers=model.centroids.detach()
        #flags shows if all clusters will be partioned later on
        split_flag=1
        global_cl=torch.zeros(self.latent_z.shape[0]).long()
        initial_mask=torch.arange(self.latent_z.shape[0])
        init_id=0
        sum_leafs=0
        if self.missing_data:
            self.missing_data_positions(cl_idx,sparse_mask,first_layer=True)

        #self.thetas.append(theta_approx)
        for i in range(depth):
            if i==self.max_layers:
                self.cond_control=10*self.cond_control
                print(self.cond_control)
            if i>29:
                print(i)
        #datapoints in each of the corresponding clusters
            #Compute points in each cluster
            assigned_points= torch.zeros(sparse_mask.shape[0])
            assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            #Splitting criterion decides if a cluster has enough points to be binary splitted
            self.splitting_criterion=(assigned_points>self.minimum_points)
            #print(assigned_points)
            #datapoints required for further splitting
            self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())

            self.mask_split=self.mask_split.squeeze(-1).bool()
            #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree
            split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
            #if not all of the clusters have enough points to be partioned
            # save them as leaf nodes and create the mask for the analytical calculations
            if not split_flag:
                #erion shows the leaf nodes
                erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
                #if we have leaf nodes in this tree level give them global ids
                if erion.sum()>0:
                    #keep the leaf nodes for visualization
                    with torch.no_grad():
                        self.leaf_centroids.append(model.centroids[erion])
                    #sum_leafs makes sure no duplicates exist in the cluster id
                    sum_leafs=sum_leafs+erion.sum()
                    
                    #global unique cluster ids to be used
                    clu_ids=torch.arange(init_id,sum_leafs)
                    #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
                    cl_vec=torch.zeros(sparse_mask.shape[0]).long()
                    cl_vec[erion]=clu_ids
                    #initial starting point for next level of ids, so no duplicates exist
                    init_id=sum_leafs
                    # print(global_cl.unique(return_counts=True))
                    # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
                    mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
                    # gl_idx takes the global node ids for the mask
                    gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
                    gl_idx=gl_idx.squeeze(-1).bool()
                    # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
                    gl_idx2=gl_idx
                    if gl_idx.shape[0]!=global_cl.shape[0]:
                        #in the case of uneven split the initial mask keeps track of the global node ids
                        gl_idx=initial_mask[gl_idx].long()

                    # give the proper cl ids to the proper nodes
                    global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
                    self.K_leafs=sum_leafs
           
            
            centers=model.centroids
            if self.splitting_criterion.sum()==0:
                break
            
            #local clusters to be splitted
            splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

            if not split_flag:
                flag_uneven_split=True
                # rename ids so they start from 0 to total splitted
                splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
                # create sparse locations of K_old x K_new
                index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
                # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
                self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask=initial_mask[self.ind[0,:].long()]
                self.mask_split=initial_mask
                cl_idx=self.ind[1,:].long()
            if flag_uneven_split:
                #translate the different size of splits to the total N of the network
                self.mask_split=initial_mask
                
            
            model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions=self.latent_z[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,device=self.device)
            sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.scaling_factor.detach()*self.latent_z.detach()[self.mask_split]),self.scaling_factor*self.latent_z.detach()[self.mask_split])
            full_prev_cl=cl_idx
            #self.global_cl_likelihood_mask(global_cl)
            if cl_idx.shape[0]==self.input_size:
                self.general_mask.append(torch.arange(self.input_size))     
            else:
                self.general_mask.append(self.mask_split)
            starting_center_id=self.total_K
            self.total_K+=2*int(self.splitting_criterion.sum())            
            
            self.general_cl_id.append(cl_idx+starting_center_id)

            self.general_centroids_sub.append(model.centroids)
            
            if self.missing_data:
                self.missing_data_positions(cl_idx,sparse_mask,starting_center_id=starting_center_id,first_layer=False)
        self.global_cl_likelihood_mask(global_cl)
        return first_centers  

    def kmeans_tree_recursively(self,depth,initial_cntrs):
        self.general_mask=[]
        self.general_cl_id=[]

        self.leaf_centroids=[]
        self.general_centroids_sub=[]
        
        
        self.missing_center_i=[]
        self.missing_center_j=[]
        self.removed_bias_i=[]
        self.removed_bias_j=[]
        self.split_history=[]
        self.total_K=int(self.init_layer_split)
     

        #first iteration
        #initialize kmeans with number of clusters equal to logN
        flag_uneven_split=False
        model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= self.latent_z.shape,init_cent=initial_cntrs,device=self.device)
        sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.latent_z.detach()),self.latent_z)
        self.general_mask.append(torch.arange(self.input_size))       

        self.general_cl_id.append(cl_idx)
        self.centroids_layer1=model.centroids

        full_prev_cl=cl_idx
        first_centers=model.centroids.detach()
        #flags shows if all clusters will be partioned later on
        split_flag=1
        global_cl=torch.zeros(self.latent_z.shape[0]).long()
        initial_mask=torch.arange(self.latent_z.shape[0])
        init_id=0
        sum_leafs=0
        if self.missing_data:
            self.missing_data_positions(cl_idx,sparse_mask,first_layer=True)

        #self.thetas.append(theta_approx)
        for i in range(depth):
            if i==self.max_layers:
                self.cond_control=10*self.cond_control
                print(self.cond_control)
            if i>29:
                print(i)
        #datapoints in each of the corresponding clusters
            #Compute points in each cluster
            assigned_points= torch.zeros(sparse_mask.shape[0])
            assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            #Splitting criterion decides if a cluster has enough points to be binary splitted
            self.splitting_criterion=(assigned_points>self.minimum_points)
            #print(assigned_points)
            #datapoints required for further splitting
            self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())

            self.mask_split=self.mask_split.squeeze(-1).bool()
            #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree       
            split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
            #if not all of the clusters have enough points to be partioned
            # save them as leaf nodes and create the mask for the analytical calculations
            if not split_flag:
               
                #erion shows the leaf nodes
                erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
                #if we have leaf nodes in this tree level give them global ids
                if erion.sum()>0:
                    #keep the leaf nodes for visualization
                    with torch.no_grad():
                        self.leaf_centroids.append(model.centroids[erion])
                    #sum_leafs makes sure no duplicates exist in the cluster id
                    sum_leafs=sum_leafs+erion.sum()
                    
                    #global unique cluster ids to be used
                    clu_ids=torch.arange(init_id,sum_leafs)
                    #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
                    cl_vec=torch.zeros(sparse_mask.shape[0]).long()
                    cl_vec[erion]=clu_ids
                    #initial starting point for next level of ids, so no duplicates exist
                    init_id=sum_leafs
                    # print(global_cl.unique(return_counts=True))
                    # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
                    mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
                    # gl_idx takes the global node ids for the mask
                    gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
                    gl_idx=gl_idx.squeeze(-1).bool()
                    # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
                    gl_idx2=gl_idx
                    if gl_idx.shape[0]!=global_cl.shape[0]:
                        #in the case of uneven split the initial mask keeps track of the global node ids
                        gl_idx=initial_mask[gl_idx].long()

                    # give the proper cl ids to the proper nodes
                    global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
                    self.K_leafs=sum_leafs
           
           
            centers=model.centroids
            if self.splitting_criterion.sum()==0:
                # CHANGED THIS
                # self.cond_control=0.001
                break
            
            #local clusters to be splitted
            splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

            if not split_flag:
                flag_uneven_split=True
                # rename ids so they start from 0 to total splitted
                splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0])
                # create sparse locations of K_old x K_new
                index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
                # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
                self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1]),index_split,torch.ones(splited_cl_ids_j.shape[0]),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask=initial_mask[self.ind[0,:].long()]
                self.mask_split=initial_mask
                cl_idx=self.ind[1,:].long()
            if flag_uneven_split:
                #translate the different size of splits to the total N of the network
                self.mask_split=initial_mask
                
            # print(self.mask_split)
            # print('cl_ids',cl_idx)
            # if (i+1)<=(len(self.center_history)-1):
            #     if self.center_history[i+1]==2*int(splited_cl_ids_i.shape[0]):
            #         model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.latent_z[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=self.all_centroids[self.center_history[i]:(self.center_history[i]+self.center_history[i+1])].detach(),full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,retain_structure=True)
            #     else:
            #         model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.latent_z[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,retain_structure=False)
            # else:
            model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.latent_z[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,retain_structure=False,device=self.device)
            sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(self.latent_z.detach()[self.mask_split]),self.latent_z[self.mask_split])
            full_prev_cl=cl_idx
            if cl_idx.shape[0]==self.input_size:
                self.general_mask.append(torch.arange(self.input_size))     
            else:
                self.general_mask.append(self.mask_split)
            starting_center_id=self.total_K
            self.total_K+=2*int(self.splitting_criterion.sum())
            self.general_cl_id.append(cl_idx+starting_center_id)
            self.general_centroids_sub.append(model.centroids)
            if self.missing_data:
                self.missing_data_positions(cl_idx,sparse_mask,starting_center_id=starting_center_id,first_layer=False)
        self.global_cl_likelihood_mask(global_cl)

        return first_centers
        
    def global_cl_likelihood_mask(self,global_cl):
        '''
        Returns the indexes of the mask required for the analytical evaluations of the last layer of the tree
        
        '''
        #make it to start from zero rather than 1
        # indexA=sparse_mask._indices()
        # values=sparse_mask._values()
        # indexB=sparse_mask.transpose(0,1)._indices()
        N_values=torch.arange(global_cl.shape[0])
        indices_N_K=torch.cat([N_values.unsqueeze(0),global_cl.unsqueeze(0)],0)
        values=torch.ones(global_cl.shape[0])
        self.global_cl=global_cl
        # if matrices are not coalesced it does not work
        #Enforce it with 'coalesced=True'

        indexC, valueC = spspmm(indices_N_K,values,indices_N_K[[1,0]],values,global_cl.shape[0],self.K_leafs,global_cl.shape[0],coalesced=True)
        mask_leafs=indexC[0]<indexC[1]
        self.analytical_i=indexC[0][mask_leafs]
        self.analytical_j=indexC[1][mask_leafs]
        
        
        
          
          
          
    
    def missing_data_positions(self,cluster_idx,sparse_mask,starting_center_id=None,first_layer=False):
            
        if first_layer:
            center_info_i=cluster_idx[self.removed_i]
            center_info_j=cluster_idx[self.removed_j]
                   
            mask_centers=center_info_i!=center_info_j
           
            self.first_missing_center_i=center_info_i[mask_centers]
            self.first_missing_center_j=center_info_j[mask_centers]
            self.removed_bias_i.append(self.removed_i[mask_centers])
            self.removed_bias_j.append(self.removed_j[mask_centers])

                    
                
        else:
            if self.mask_split.shape[0]<self.input_size:
                # prepare mask to translate missing connections to uneven splits
                # N'xN NxN NxN'
                indices_translator=torch.cat([torch.arange(self.mask_split.shape[0]).unsqueeze(0),self.mask_split.unsqueeze(0)],0)
                missing_connections=torch.cat([self.removed_i.unsqueeze(0),self.removed_j.unsqueeze(0)],0)
                indexC, valueC = spspmm(missing_connections,torch.ones(missing_connections.shape[1]), indices_translator[[1,0]],torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.mask_split.shape[0],coalesced=True)
                if valueC.shape[0]==0:
                    return
                indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.mask_split.shape[0],self.input_size,self.mask_split.shape[0],coalesced=True)
                     
                center_info_i=cluster_idx[indexC[0]]
                center_info_j=cluster_idx[indexC[1]]
                #calculate minimum even centroids to distinguish which ones belong on the same branch
                binary_mask_even_minimum=(torch.minimum(center_info_i, center_info_j)%2)==0
                binary_mask_branch_mask=torch.abs(center_info_i-center_info_j)==1
                final_mask_keep= binary_mask_even_minimum*binary_mask_branch_mask

                center_info_i=center_info_i[final_mask_keep]
                center_info_j=center_info_j[final_mask_keep]
                        
                indexC, valueC = spspmm(missing_connections,(missing_connections[1]+1).float(), indices_translator[[1,0]],torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.mask_split.shape[0],coalesced=True)
                indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.mask_split.shape[0],self.input_size,self.mask_split.shape[0],coalesced=True)
                removed_j_from_og=(valueC-1).long()
                        
                indexC, valueC = spspmm(missing_connections,(missing_connections[0]+1).float(), indices_translator[[1,0]],torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.mask_split.shape[0],coalesced=True)
                indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.mask_split.shape[0],self.input_size,self.mask_split.shape[0],coalesced=True)
                removed_i_from_og=(valueC-1).long()
                       
                        
                self.missing_center_i.append(center_info_i+starting_center_id)
                self.missing_center_j.append(center_info_j+starting_center_id)
                self.removed_bias_i.append(removed_i_from_og[final_mask_keep])
                self.removed_bias_j.append(removed_j_from_og[final_mask_keep])
            else:
                        
                center_info_i=cluster_idx[self.removed_i]
                center_info_j=cluster_idx[self.removed_j]
                #calculate minimum even centroids to distinguish which ones belong on the same branch
                binary_mask_even_minimum=(torch.minimum(center_info_i, center_info_j)%2)==0
                binary_mask_branch_mask=torch.abs(center_info_i-center_info_j)==1
                final_mask_keep= binary_mask_even_minimum*binary_mask_branch_mask
                        
                        

                        
                #clusters to keep
                center_info_i=center_info_i[final_mask_keep]
                center_info_j=center_info_j[final_mask_keep]
        
                           
                self.missing_center_i.append(center_info_i+starting_center_id)
                self.missing_center_j.append(center_info_j+starting_center_id)
                self.removed_bias_i.append(self.removed_i[final_mask_keep])
                self.removed_bias_j.append(self.removed_j[final_mask_keep])

                       
                        


    
    def pairwise_squared_distance_trick(self,X,epsilon):
        '''
        Calculates the pairwise distance of a tensor in a memory efficient way
        
        '''
        Gram=torch.mm(X,torch.transpose(X,0,1))
        dist=torch.diag(Gram,0).unsqueeze(0)+torch.diag(Gram,0).unsqueeze(-1)-2*Gram+epsilon
        return dist
    
                    

                    