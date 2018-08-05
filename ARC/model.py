from ARC import * 
import  matplotlib.pyplot as plt 

class ARC(nn.Module):
    def __init__(self, batch_size, CauchyKerSize, hidden_size, num_glimpses  ):
        super(ARC, self).__init__()
        self.img_size = 32
        self.batch_size = batch_size
        self.cauchy_ker_size = CauchyKerSize
        self.num_glimpses = num_glimpses
        self.hidden_size = hidden_size

        # grids for computing cauchy kernels (attenttion) 
        self.cauchy_kernel_grid = torch.arange( self.cauchy_ker_size, dtype = torch.float ) - ( self.cauchy_ker_size / 2 + 0.5 )
        self.filterbank_grid = torch.arange(self.img_size, dtype = torch.float)[None, None, :].expand(
            self.batch_size, self.cauchy_ker_size, -1)
        
        # ARC implementation: outchannels = 16, kernel_size = (3,3), stride = 1  
        self.channels = 16
        self.convolve = nn.Conv2d(1, self.channels, 
        padding = (1,1), 
        kernel_size = (3,3), stride = 1 ) 
        
        self.attn_patch_size = self.channels * self.cauchy_ker_size * self.cauchy_ker_size
        self.controller = nn.GRUCell(
            input_size=self.attn_patch_size, hidden_size= self.hidden_size
        ) 

        # W_g in the paper, projects hidden state of controller to (x_hat, y_hat, delta_hat)
        self.hidden_projection = nn.Linear(self.hidden_size, 3)

        # maps the final hidden state to raw similarity scores 
        self.linear_output = nn.Linear(self.hidden_size, 64)
        self.linear_output2 = nn.Linear(64, 1)

    def _compute_filterbanks(self, z, img_axis_center, gamma):
        ''' 
        Attention parameters are computed with processed info from current hidden state h_t and the image
        Args:
            z (batch size, 1): Cauchy kernel center 
            img_axis_center (batch size, ):
            gamma (B, 1):
        '''
        mu_Z = z + img_axis_center #(64, 1) (64, 4, 1)
        
        F_Z = self.filterbank_grid - mu_Z[:,:,None]
        F_Z = 1.0 + (( F_Z / gamma[:, None, None] )**2)
        F_Z = 1.0 /( np.pi* F_Z * gamma[:, None, None] )
        
        F_Z = F_Z/(torch.sum(F_Z, dim = 2) + 1e-4)[:, :, None]
        F_Z = F_Z.unsqueeze(1).expand(-1, self.channels, -1, -1)
        
        return F_Z

    def batched_dot(self, A, B):
        ''' 
            Computes a tensor C that's quinessentially the high dimensional dot product
            along a common axis between A and B  
        Args:
            A (Batch size, Channels, I, J)
            B (Batch size, Channels, J, K)
        '''
        A, B = A.unsqueeze(4), B.unsqueeze(2)
        C = A * B
        return C.sum(3)
    
    def _attend(self, img, attend_params):
        # computation found in section 2.1
        x_hat, y_hat  = attend_params[:, 0][:, None], attend_params[:, 1][:, None]
        delta_hat = torch.abs( attend_params[:, 2] )
        
        x, y = (self.img_size - 1)*(x_hat+1)/2.0, (self.img_size - 1)*(y_hat+1)/2.0 
        delta = self.img_size * (1.0 - delta_hat) / self.cauchy_ker_size
        gamma = torch.exp( 1-( 2.0 * delta_hat ) )
        
        img_axis_center = delta[:,None] * self.cauchy_kernel_grid[None,:]  # elementwise mult w/ delta
        
        F_X = self._compute_filterbanks(x, img_axis_center, gamma) 
        F_Y = self._compute_filterbanks(y, img_axis_center, gamma)
        
        #print(F_Y.shape, img.shape, F_X.shape, attn_patch.shape)
        #[batch, channel, N, S] [batch, channel, S, S] [batch, channel, S, N] 
        
        # Attend to image: Apply Cauchy kernel to the current image to extract a patch 
        attn_patch = self.batched_dot( F_Y, img)
        attn_patch = self.batched_dot(attn_patch , torch.transpose(F_X, 2, 3) )
        
        return attn_patch

    def glimpse(self, img, hidden):        
        # compute  attention parameters
        attn_params = torch.tanh( self.hidden_projection( hidden ) )
        
        # extract relevant patch of the drawing
        attn_patch = self._attend(img, attn_params)
        attn_patch = attn_patch.reshape(self.batch_size, -1)
        
        # update the RNN hidden state h_t
        hidden = self.controller( attn_patch, hidden )
        return hidden

    def forward(self, X, ind = None):        
        # apply convolution to both support and test drawings, adds depth dimension
        support = F.elu( self.convolve( X[:, 0, :, :].unsqueeze(1) ) )
        test = F.elu( self.convolve( X[:, 1, :, :].unsqueeze(1) ) )
        
        # init RNN (controller)
        hidden = torch.zeros(self.batch_size, self.hidden_size)
        
        glimpse_stats = np.zeros( (self.batch_size, 8) )
        for i in range(8): # times glimpsing support-test pairs
            hidden = self.glimpse( support, hidden )
            hidden = self.glimpse( test, hidden )

            # compute glimpse score for analysis
            glimpse_score = F.elu( self.linear_output2( self.linear_output( hidden ) ) ) 
            glimpse_stats[:, i] = glimpse_score.reshape(-1).detach().numpy()

        # after glimpsing both images, create a raw score for similarity
        hidden = F.elu( self.linear_output2( self.linear_output( hidden ) ) )
        
        if ind:
            np.save('./data/glimpse_stats/%s' %ind, glimpse_stats)
        return hidden 
