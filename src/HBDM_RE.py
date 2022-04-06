import torch
import torch.nn as nn
import numpy as np
import torch_sparse
from fractal_main_cond import Tree_kmeans_recursion
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# device = torch.device("cpu")
# torch.set_default_tensor_type('torch.FloatTensor')
undirected=1





class LSM(nn.Module,Tree_kmeans_recursion,Spectral_clustering_init):
    def __init__(self,data,sparse_i,sparse_j, input_size,latent_dim,graph_type,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=False,initialization=None,scaling=None,missing_data=False,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),LP=True):
        super(LSM, self).__init__()
        Tree_kmeans_recursion.__init__(self,minimum_points=3*int(data.shape[0]/(data.shape[0]/np.log(data.shape[0]))),init_layer_split=3*torch.round(torch.log(torch.tensor(data.shape[0]).float())))
        Spectral_clustering_init.__init__(self,device=device)
        self.input_size=input_size
        self.cluster_evolution=[]
        self.mask_evolution=[]
        self.init_layer_split=torch.round(torch.log(torch.tensor(data.shape[0]).float()))
        self.init_layer_idx=torch.triu_indices(int(self.init_layer_split),int(self.init_layer_split),1)
       
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        self.initialization=1
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))
        self.build_hierarchy=False
        self.graph_type=graph_type
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.flag1=0
        self.sparse_j_idx=sparse_j
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        self.CUDA=True
        self.pdist_tol1=nn.PairwiseDistance(p=2,eps=0)

        self.device=device
        
        if LP:
            self.non_sparse_i_idx_removed=non_sparse_i
         
            self.non_sparse_j_idx_removed=non_sparse_j
               
            self.sparse_i_idx_removed=sparse_i_rem
            self.sparse_j_idx_removed=sparse_j_rem
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        
        self.spectral_data=self.spectral_clustering()#.flip(1)

        self.first_centers_sp=torch.randn(int(self.init_layer_split),self.spectral_data.shape[1],device=device)

        global_cl,spectral_leaf_centers=self.kmeans_tree_z_initialization(depth=80,initial_cntrs=self.first_centers_sp) 
           
        self.first_centers=torch.randn(int(torch.round(torch.log(torch.tensor(data.shape[0]).float()))),latent_dim,device=device)
      

        spectral_centroids_to_z=spectral_leaf_centers[global_cl]
        # spectral_centroids_to_z=self.spectral_data
        if self.spectral_data.shape[1]>latent_dim:
            self.latent_z=nn.Parameter(spectral_centroids_to_z[:,0:latent_dim])
        elif self.spectral_data.shape[1]==latent_dim:
            self.latent_z=nn.Parameter(spectral_centroids_to_z)
        else:
            self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
            self.latent_z.data[:,0:self.spectral_data.shape[1]]=spectral_centroids_to_z
   
    def approximation_blocks(self):
        if self.scaling:
            # UPDATE the fractal structure some epochs
            if self.build_hierarchy:
                self.center_history=[]

                self.build_hierarchy=False
                # centroids of the kmean procedure
                self.first_centers=torch.randn(int(self.init_layer_split),self.latent_dim,device=self.device)
                init_centroids=self.kmeans_tree_scaling(depth=80,initial_cntrs=self.first_centers)
                self.first_centers=init_centroids
                #total number of centroids of the hierarchy
                self.center_history=[]

                #cluster assignemnts based on the whole fractal structure
                self.k_i=torch.cat(self.general_cl_id)
                self.N_j=torch.cat(self.general_mask)
                # calculate total NxK spase mask of cluster allocation
                self.general_matrix=torch.sparse.FloatTensor(torch.cat((self.k_i.unsqueeze(0),self.N_j.unsqueeze(0)),0),torch.ones(self.N_j.shape[0]),torch.Size([self.total_K,self.input_size]))
                # pairwise distances of the first layer
                k_distance_first_layer=torch.exp(-torch.pdist(self.centroids_layer1))
                #pairwise distances of the subsequent centroids
                total_centroids=torch.cat(self.general_centroids_sub)
                for h_i in range(len(self.general_centroids_sub)):
                    self.center_history.append(self.general_centroids_sub[h_i].shape[0])
                self.all_centroids=torch.cat((self.centroids_layer1,total_centroids))
            
                
                # create the KxK  distance matrix
                k_distance_sub=torch.exp(-self.pdist(total_centroids.view(-1,2,total_centroids.shape[-1])[:,0,:],total_centroids.view(-1,2,total_centroids.shape[-1])[:,1,:]))
                sparse_k_idx=torch.arange(self.init_layer_split,self.init_layer_split+total_centroids.shape[0]).long().view(-1,2).transpose(0,1)
                self.final_idx=torch.cat((self.init_layer_idx,sparse_k_idx),1)
                self.k_exp_dist=torch.sparse.FloatTensor(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),torch.Size([self.total_K,self.total_K]))

                
                # calculate bias x mask
                sum_cl_idx=torch.sparse.mm(self.general_matrix,torch.exp(self.gamma).unsqueeze(-1))
                # translate mask positions to distance positions
                if self.missing_data:
                    # create triangular matrix for distance translation of first layer centroids
                    self.translate_idx_to_distance_pos=torch.sparse.FloatTensor(self.init_layer_idx,torch.arange(self.init_layer_idx.shape[-1]),torch.Size([int(self.init_layer_split),int(self.init_layer_split)]))
                    self.translate_idx_to_distance_pos=(self.translate_idx_to_distance_pos+self.translate_idx_to_distance_pos.transpose(0,1)).to_dense()
                    first_missing_centers=self.translate_idx_to_distance_pos[self.first_missing_center_i,self.first_missing_center_j]
                    sub_missing_centers=(torch.minimum(torch.cat(self.missing_center_i)-self.init_layer_split, torch.cat(self.missing_center_j)-self.init_layer_split)/2).long()+self.init_layer_idx.shape[-1]
                    self.total_missing=torch.cat((first_missing_centers,sub_missing_centers))
                    self.removed_bias_i=torch.cat(self.removed_bias_i)
                    self.removed_bias_j=torch.cat(self.removed_bias_j)
                    
                    mask_extra=self.global_cl[self.removed_i]==self.global_cl[self.removed_j]
                    self.extra_i=self.removed_i[mask_extra]
                    self.extra_j=self.removed_j[mask_extra]
                    
                   
                    
            # KEEPS the fractal structure as it is 
            else:
                for i in range(100):
                    if i==0:
                        with torch.no_grad():
                            self.all_centroids=self.update_clusters_local()
                            previous_centers=self.all_centroids
                    else:
                        with torch.no_grad():
                            self.all_centroids=self.update_clusters_local()
                        if i%5==0:
                            if self.pdist_tol1(previous_centers,self.all_centroids).sum()<1e-3:
                                break
                        previous_centers=self.all_centroids.detach()
                self.all_centroids=self.update_clusters_local()
                self.general_matrix=torch.sparse.FloatTensor(torch.cat((self.k_i.unsqueeze(0),self.N_j.unsqueeze(0)),0),torch.ones(self.N_j.shape[0]),torch.Size([self.total_K,self.input_size]))

                
                self.centroids_layer1=self.all_centroids[0:int(self.init_layer_split)]
                self.first_centers=self.centroids_layer1.detach()

                k_distance_first_layer=torch.exp(-torch.pdist(self.centroids_layer1))
                #pairwise distances of the subsequent centroids
                total_centroids=self.all_centroids[int(self.init_layer_split):]
                    
                
                # create the KxK  distance matrix
                k_distance_sub=torch.exp(-self.pdist(total_centroids.view(-1,2,total_centroids.shape[-1])[:,0,:],total_centroids.view(-1,2,total_centroids.shape[-1])[:,1,:]))
                self.k_exp_dist=torch.sparse.FloatTensor(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),torch.Size([self.total_K,self.total_K]))

                # calculate bias x mask
                sum_cl_idx=torch.sparse.mm(self.general_matrix,torch.exp(self.gamma).unsqueeze(-1))
    
           
            if self.missing_data:
                theta_missing=(torch.exp(self.gamma[self.removed_bias_i])*torch.exp(self.gamma[self.removed_bias_j])* torch.cat((k_distance_first_layer,k_distance_sub))[self.total_missing]).sum()
               # torch_sparse.spmm(indices, values, n, n, Y_dense)
                theta_approx=torch.exp(self.bias)*(torch.mm(sum_cl_idx.transpose(0,1),(torch_sparse.spmm(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),self.total_K,self.total_K,sum_cl_idx))).sum())-torch.exp(self.bias)*theta_missing

                #theta_approx=torch.exp(self.bias)*(torch.mm(sum_cl_idx.transpose(0,1),(torch.sparse.mm(self.k_exp_dist,sum_cl_idx))).sum())-torch.exp(self.bias)*theta_missing
                self.theta_approx=theta_approx
            else:
                # calculat approximation
                theta_approx=torch.exp(self.bias)*(torch.mm(sum_cl_idx.transpose(0,1),(torch_sparse.spmm(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),self.total_K,self.total_K,sum_cl_idx))).sum())

        else:
            if self.build_hierarchy:

                self.build_hierarchy=False
                # update fractal structure
                self.first_centers=torch.randn(int(self.init_layer_split),self.latent_dim,device=self.device)

                init_centroids=self.kmeans_tree_recursively(depth=80,initial_cntrs=self.first_centers)
                self.first_centers=init_centroids
                self.center_history=[]

                self.k_i=torch.cat(self.general_cl_id)
                self.N_j=torch.cat(self.general_mask)
                # calculate total NxK spase mask of cluster allocation
                self.general_matrix=torch.sparse.FloatTensor(torch.cat((self.k_i.unsqueeze(0),self.N_j.unsqueeze(0)),0),torch.ones(self.N_j.shape[0]),torch.Size([self.total_K,self.input_size]))
                # pairwise distances of the first layer
                k_distance_first_layer=torch.exp(-torch.pdist(self.centroids_layer1))
                #pairwise distances of the subsequent centroids
                total_centroids=torch.cat(self.general_centroids_sub)
                self.center_history.append(init_centroids.shape[0])
                for h_i in range(len(self.general_centroids_sub)):
                    self.center_history.append(self.general_centroids_sub[h_i].shape[0])
                
                self.all_centroids=torch.cat((self.centroids_layer1,total_centroids))
            
                
                # create the KxK  distance matrix
                k_distance_sub=torch.exp(-self.pdist(total_centroids.view(-1,2,total_centroids.shape[-1])[:,0,:],total_centroids.view(-1,2,total_centroids.shape[-1])[:,1,:]))
                sparse_k_idx=torch.arange(self.init_layer_split,self.init_layer_split+total_centroids.shape[0]).long().view(-1,2).transpose(0,1)
                self.final_idx=torch.cat((self.init_layer_idx,sparse_k_idx),1)
                self.k_exp_dist=torch.sparse.FloatTensor(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),torch.Size([self.total_K,self.total_K]))
                
                # calculate bias x mask
                sum_cl_idx=torch.sparse.mm(self.general_matrix,torch.exp(self.gamma).unsqueeze(-1))
                # update centroids
                if self.missing_data:
                    # create triangular matrix for distance translation of first layer centroids
                    self.translate_idx_to_distance_pos=torch.sparse.FloatTensor(self.init_layer_idx,torch.arange(self.init_layer_idx.shape[-1]),torch.Size([int(self.init_layer_split),int(self.init_layer_split)]))
                    self.translate_idx_to_distance_pos=(self.translate_idx_to_distance_pos+self.translate_idx_to_distance_pos.transpose(0,1)).to_dense()
                    first_missing_centers=self.translate_idx_to_distance_pos[self.first_missing_center_i,self.first_missing_center_j]
                    sub_missing_centers=(torch.minimum(torch.cat(self.missing_center_i)-self.init_layer_split, torch.cat(self.missing_center_j)-self.init_layer_split)/2).long()+self.init_layer_idx.shape[-1]
                    self.total_missing=torch.cat((first_missing_centers,sub_missing_centers))
                    self.removed_bias_i=torch.cat(self.removed_bias_i)
                    self.removed_bias_j=torch.cat(self.removed_bias_j)
                    
                    mask_extra=self.global_cl[self.removed_i]==self.global_cl[self.removed_j]
                    self.extra_i=self.removed_i[mask_extra]
                    self.extra_j=self.removed_j[mask_extra]
            else:
               
                for i in range(100):
                    if i==0:
                        with torch.no_grad():
                            self.all_centroids=self.update_clusters_local()
                            previous_centers=self.all_centroids
                    else:
                        with torch.no_grad():
                            self.all_centroids=self.update_clusters_local()
                        if i%5==0:
                            if self.pdist_tol1(previous_centers,self.all_centroids).sum()<1e-4:
                                break
                        previous_centers=self.all_centroids.detach()
                self.all_centroids=self.update_clusters_local()
                self.general_matrix=torch.sparse.FloatTensor(torch.cat((self.k_i.unsqueeze(0),self.N_j.unsqueeze(0)),0),torch.ones(self.N_j.shape[0]),torch.Size([self.total_K,self.input_size]))

                
                self.centroids_layer1=self.all_centroids[0:int(self.init_layer_split)]
                self.first_centers=self.centroids_layer1.detach()
                k_distance_first_layer=torch.exp(-torch.pdist(self.centroids_layer1))
                #pairwise distances of the subsequent centroids
                total_centroids=self.all_centroids[int(self.init_layer_split):]
                    
                
                # create the KxK  distance matrix
                k_distance_sub=torch.exp(-self.pdist(total_centroids.view(-1,2,total_centroids.shape[-1])[:,0,:],total_centroids.view(-1,2,total_centroids.shape[-1])[:,1,:]))
                self.k_exp_dist=torch.sparse.FloatTensor(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),torch.Size([self.total_K,self.total_K]))

                
                # calculate bias x mask
                sum_cl_idx=torch.sparse.mm(self.general_matrix,torch.exp(self.gamma).unsqueeze(-1))
    
           
            
            if self.missing_data:
                theta_missing=(torch.exp(self.gamma[self.removed_bias_i])*torch.exp(self.gamma[self.removed_bias_j])* torch.cat((k_distance_first_layer,k_distance_sub))[self.total_missing]).sum()
               # theta_approx=(torch.mm(sum_cl_idx.transpose(0,1),(torch.sparse.mm(self.k_exp_dist,sum_cl_idx))).sum())-theta_missing
                theta_approx=(torch.mm(sum_cl_idx.transpose(0,1),(torch_sparse.spmm(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),self.total_K,self.total_K,sum_cl_idx))).sum())-theta_missing

            else:
                # calculate approximation
                theta_approx=(torch.mm(sum_cl_idx.transpose(0,1),(torch_sparse.spmm(self.final_idx,torch.cat((k_distance_first_layer,k_distance_sub)),self.total_K,self.total_K,sum_cl_idx))).sum())
            
            
            
            
        return theta_approx
 
                  
  
    
    def update_clusters_local(self):
       
        z = torch.zeros(self.total_K, self.latent_dim,device=self.device)
        o = torch.zeros(self.total_K,device=self.device)
        
       
        if self.scaling:
            with torch.no_grad():
                
                lambdas_full= (((self.scaling_factor*self.latent_z[self.N_j]-self.all_centroids[self.k_i])**2).sum(-1))**0.5+1e-06    
                inv_lambdas=1/lambdas_full
            self.lambdas_X=torch.mul(self.scaling_factor*self.latent_z[self.N_j].detach(),inv_lambdas.unsqueeze(-1))
        else:
            with torch.no_grad():
                
                lambdas_full= (((self.latent_z[self.N_j]-self.all_centroids[self.k_i])**2).sum(-1))**0.5+1e-06                
                inv_lambdas=1/lambdas_full
            self.lambdas_X=torch.mul(self.latent_z[self.N_j],inv_lambdas.unsqueeze(-1))
            self.inv_lambdas=inv_lambdas
     
       
        z=z.index_add(0, self.k_i, self.lambdas_X)
        o=o.index_add(0, self.k_i, inv_lambdas)

        centroids=torch.mul(z,(1/(o+1e-06)).unsqueeze(-1))
        return centroids
        
        
        


    def local_likelihood(self):
        '''

        Parameters
        ----------
        k_mask : data points belonging to the specific centroid

        Returns
        -------
        Explicit calculation over the box of a specific centroid

        '''
     
        if self.missing_data:
            
            
            if self.scaling:
                
                block_pdist=self.pdist(self.scaling_factor*self.latent_z[self.analytical_i].detach(),self.scaling_factor*self.latent_z[self.analytical_j].detach())+1e-08
                
                extra_block_pdist=self.pdist(self.scaling_factor*self.latent_z[self.extra_i].detach(),self.scaling_factor*self.latent_z[self.extra_j].detach())+1e-08
           
            
    ## Block kmeans analytically#########################################################################################################
            
                lambda_block=-block_pdist+self.gamma[self.analytical_i]+self.gamma[self.analytical_j]+self.bias
            
                extra_lambda_block=-extra_block_pdist+self.gamma[self.extra_i]+self.gamma[self.extra_j]+self.bias
                an_lik=torch.exp(lambda_block).sum()-torch.exp(extra_lambda_block).sum()
            else:
                block_pdist=self.pdist(self.latent_z[self.analytical_i],self.latent_z[self.analytical_j])+1e-08
                
                extra_block_pdist=self.pdist(self.latent_z[self.extra_i],self.latent_z[self.extra_j])+1e-08
           
            
    ## Block kmeans analytically#########################################################################################################
            
                lambda_block=-block_pdist+self.gamma[self.analytical_i]+self.gamma[self.analytical_j]
            
                extra_lambda_block=-extra_block_pdist+self.gamma[self.extra_i]+self.gamma[self.extra_j]
                an_lik=torch.exp(lambda_block).sum()-torch.exp(extra_lambda_block).sum()
        
        else:
            if self.scaling:
                block_pdist=self.pdist(self.scaling_factor*self.latent_z[self.analytical_i].detach(),self.scaling_factor*self.latent_z[self.analytical_j].detach())+1e-08
                
                ## Block kmeans analytically#########################################################################################################
                
                lambda_block=-block_pdist+self.gamma[self.analytical_i]+self.gamma[self.analytical_j]+self.bias
                an_lik=torch.exp(lambda_block).sum()
                
            else:
                block_pdist=self.pdist(self.latent_z[self.analytical_i],self.latent_z[self.analytical_j])+1e-08
             
                ## Block kmeans analytically#########################################################################################################
                
                lambda_block=-block_pdist+self.gamma[self.analytical_i]+self.gamma[self.analytical_j]
                an_lik=torch.exp(lambda_block).sum()

        return an_lik
        
    
    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias(self,epoch):
        '''

        Parameters
        ----------
        cent_dist : real
            distnces of the updated centroid and the k-1 other centers.
        count_prod : TYPE
            DESCRIPTION.
        mask : Boolean
            DESCRIBES the slice of the mask for the specific kmeans centroid.

        Returns
        -------
        None.

        '''
        self.epoch=epoch
        
        if self.missing_data:
            if self.scaling:
                thetas=self.approximation_blocks()
                #theta_stack=torch.stack(self.thetas).sum()
                analytical_blocks_likelihood=self.local_likelihood()
         ##############################################################################################################################
                
                z_pdist=(((self.scaling_factor*self.latent_z[self.sparse_i_idx].detach()-self.scaling_factor*self.latent_z[self.sparse_j_idx].detach()+1e-06)**2).sum(-1))**0.5
                
                z_pdist_missing=(((self.scaling_factor*self.latent_z[self.sparse_i_idx_removed].detach()-self.scaling_factor*self.latent_z[self.sparse_j_idx_removed].detach()+1e-06)**2).sum(-1))**0.5
                logit_u_missing=-z_pdist_missing+self.gamma[self.sparse_i_idx_removed]+self.gamma[self.sparse_j_idx_removed]+self.bias


                
        ####################################################################################################################################
               
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]+self.bias
         #########################################################################################################################################################      
                log_likelihood_sparse=torch.sum(logit_u)-thetas-(analytical_blocks_likelihood)-torch.sum(logit_u_missing)
                if self.epoch==500:
                    self.gamma.data=0.5*self.bias.data+self.gamma.data
                    self.scaling=0
                    self.first_centers=self.first_centers*self.scaling_factor.data
                    self.latent_z.data=self.latent_z.data*self.scaling_factor.data
                
                
                
            else:
                thetas=self.approximation_blocks()
                #theta_stack=torch.stack(self.thetas).sum()
                analytical_blocks_likelihood=self.local_likelihood()
         ##############################################################################################################################
                
                z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_z[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
                
        ####################################################################################################################################
                z_pdist_missing=(((self.latent_z[self.sparse_i_idx_removed]-self.latent_z[self.sparse_j_idx_removed]+1e-06)**2).sum(-1))**0.5
                logit_u_missing=-z_pdist_missing+self.gamma[self.sparse_i_idx_removed]+self.gamma[self.sparse_j_idx_removed]
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
               
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]
         #########################################################################################################################################################      
                log_likelihood_sparse=torch.sum(logit_u)-thetas-analytical_blocks_likelihood-torch.sum(logit_u_missing)
        #############################################################################################################################################################        
            
        else:
            if self.scaling:
                thetas=self.approximation_blocks()
                #theta_stack=torch.stack(self.thetas).sum()
                analytical_blocks_likelihood=self.local_likelihood()
         ##############################################################################################################################
               
                z_pdist=(((self.scaling_factor*self.latent_z[self.sparse_i_idx].detach()-self.scaling_factor*self.latent_z[self.sparse_j_idx].detach()+1e-06)**2).sum(-1))**0.5
                
        ####################################################################################################################################
                
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
               
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]+self.bias
         #########################################################################################################################################################      
                log_likelihood_sparse=torch.sum(logit_u)-thetas-(analytical_blocks_likelihood)
                if self.epoch==500:
                    self.gamma.data=0.5*self.bias.data+self.gamma.data
                    self.scaling=0
                    self.first_centers=self.first_centers*self.scaling_factor.data
                    self.latent_z.data=self.latent_z.data*self.scaling_factor.data
                
                
                
            else: 
               
                thetas=self.approximation_blocks()

               
                analytical_blocks_likelihood=self.local_likelihood()
             
                z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_z[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
             
        ####################################################################################################################################
                
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
               
                logit_u=-z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]

         #########################################################################################################################################################      
                log_likelihood_sparse=torch.sum(logit_u)-thetas-analytical_blocks_likelihood
               
        #############################################################################################################################################################        
                 
            
        return log_likelihood_sparse
    
    
   
    
    
    
    def link_prediction(self):
        with torch.no_grad():
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j])**2).sum(-1))**0.5
            logit_u_miss=-z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=torch.exp(logit_u_miss)

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(recall,precision)
    
    
   
   
    
