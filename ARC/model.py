from ARC import * 

# meta_data["training_loss"], meta_data["validation_loss"] = [], []
# meta_data["validation_accuracy"] = []

class ARC(nn.Module):
    def __init__(self, batch_size):
        super(ARC, self).__init__()
        self.batch_size = batch_size

        # ARC implementation: outchannels = 16, kernel_size = (3,3), stride = 1  
        self.channels = 4
        self.convolve = nn.Conv2d(1, self.channels, kernel_size = (3,3), stride = 1 ) 
        self.relu = nn.ReLU()
        
        self.hidden_size = 576 #35*35*2
        self.controller = nn.GRU(self.hidden_size, self.hidden_size, batch_first = True) 

        # W_g in the paper, projects hidden state of controller to (x_hat, y_hat, delta_hat)
        self.hidden_projection = nn.Linear(self.hidden_size, 3)
            
        self.linear_output = nn.Linear(self.hidden_size, 2)
    
    def _compute_filterbanks(self, z, delta, gamma, S = 30, N = 12):
        mu_Z = np.array(  [ z + (i - (((N+1)/2) ) )*delta for i in range(N)   ] )
        F_Z = []
        for pixel_index in range(S):
            F_Z.append( 1+( (pixel_index - mu_Z)/gamma )**2 )

        F_Z = 1/( np.pi*gamma*np.array(F_Z) )
        F_Z = F_Z.T
        
        F_Z = torch.from_numpy(F_Z)
        return F_Z

    def _attend(self, img, attend_params, S = 30, N = 12):
        x_hat, y_hat, delta_hat = attend_params[:, 0], attend_params[:, 1], attend_params[:, 2]

        x, y = (S - 1)*(x_hat+1)/2, (S - 1)*(y_hat+1)/2 
        delta = S * (1-np.abs(delta_hat)) / N
        gamma = np.exp( 1-(2*np.abs(delta_hat) ) )

        F_X, F_Y = self._compute_filterbanks(x, delta, gamma), self._compute_filterbanks(y, delta, gamma)
        F_Y = torch.transpose(F_Y, 1, 2)

        attn_patch = torch.zeros( 8, self.channels, N, N)
        for c in range(self.channels):
            attn_patch[:, c, :,: ] = torch.bmm( torch.bmm(F_X, img[:, c, :, :]), F_Y) 
                
        return attn_patch #torch.from_numpy( attn_patch)

    def glimpse(self, img, hidden):
        attn_params = self.hidden_projection(hidden)
        attn_params = attn_params.detach().squeeze(0) 
        #print(attn_params  )

        attn_patch = self._attend(img, attn_params)
        attn_patch = attn_patch.reshape(attn_patch.shape[0], 1 , -1)
        _, hidden = self.controller( attn_patch, hidden )
        return  hidden

    def _conv(self, img):
        img = self.relu( self.convolve(img) )
        return img

    def forward(self, support, test):
        support = self._conv(support)
        test = self._conv(test)
        
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)

        for i in range(6): # num glimpses
            if i % 2 == 0:
                hidden = self.glimpse( support, hidden )

            else: 
                hidden = self.glimpse( test, hidden )

        hidden = hidden.squeeze(0)#reshape(4, -1)
        hidden = self.linear_output( hidden )
        return hidden 
