import numpy as np
from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.stats import f

def parameters(r, p, q, d, H_0=True):
    beta = np.random.uniform(0, 1, (r, p))
    beta = beta/np.linalg.norm(beta)
    gamma = np.random.uniform(0, 1, (p, q))
    gamma = gamma/np.linalg.norm(gamma)
    A_x, A_z = np.random.uniform(0, 1, (q, d)), np.random.uniform(0, 1, (r, d))
    A_x, A_z = A_x/np.linalg.norm(A_x), A_z/np.linalg.norm(A_z)
    if H_0 :
        A_x = np.random.uniform(0, 0, (q, d))
    return A_x, A_z, beta, gamma

def generate_psd_block_diagonal_matrix(d, block_size=10):
    """
    Generates a block diagonal matrix of size d x d where each block is d/10 x d/10 and the matrix is PSD.
    
    Parameters:
    - d: Dimension of the resulting square matrix. Must be divisible by 10.
    
    Returns:
    - A PSD block diagonal matrix of size d x d.
    """
    if d % block_size != 0:
        raise ValueError("d must be divisible by 10")
    
    num_blocks = d // block_size
    
    blocks = []
    for _ in range(num_blocks):
        # Generate a random symmetric matrix
        A = np.random.rand(block_size, block_size)
        symmetric_matrix = (A + A.T) / 2
        
        # Ensure the matrix is PSD by making it a covariance matrix
        psd_matrix = symmetric_matrix @ symmetric_matrix.T
        
        blocks.append(psd_matrix)
    
    # Combine blocks into a block diagonal matrix
    return block_diag(*blocks)

def f_a(X, gamma=np.random.uniform(-1, 1, (5, 5)), a=1, b=2):
    # gamma = gamma/np.linalg.norm(gamma)
    Y = np.exp(-(X @ gamma) **2/b)*np.sin(a*X @ gamma)
    return Y/np.std(Y)

def f_a(X, gamma=np.random.uniform(-1, 1, (5, 5)), a=3, b=3):
    gamma = gamma/np.linalg.norm(gamma)
    Y = np.exp(-(X @ gamma) **2/b)*np.sin(a*X @ gamma)
    return Y/np.std(Y)

# def f_a(X, gamma=np.random.uniform(-1, 1, (5, 5)), a=2, b=2):
#     Y = np.cos(a*X @ gamma)
#     return Y/np.std(Y)


def generate_nonlinear_data(N, p, r, d, beta, gamma, A_x, A_z, noise_type='low_rank', rk=10, bs=10, test_size=0.05, a=0.8, b=0.1, c=0.1, nl_param=2) :
    if a + b + c != 1:
        print('(a, c, c) should be a convex combination...')
    Z = np.random.uniform(-3, 3, (N, r))
    X = f_a(Z, beta, a=nl_param) + np.random.uniform(-3, 3, (N, p))
    Y_x = X @ gamma
    if np.std(Y_x) != 0:
        Y_x = Y_x/np.std(Y_x)
    if noise_type == 'low_rank':
        A = np.random.randn(d, rk)
        Sigma = A @ A.T
    elif noise_type == 'full_rank':
        A = np.random.randn(d, d)
        Sigma = A @ A.T
    elif noise_type == 'diag':
        Sigma = np.identity(d)
    elif noise_type == 'block_diag':
        Sigma = generate_psd_block_diagonal_matrix(d, block_size=bs)
    else :
        print('Noise type not available!')

    Sigma = 0.1*Sigma/np.linalg.norm(Sigma)

    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 

    Y_z = f_a(Z, A_z, a=nl_param)

    
    Y = a * Y_x @ A_x + b *  Y_z + c*N_Y

    return X, Y, Z, Y_x

def generate_data(N, p, r, d, beta, gamma, A_x, A_z, noise_type='low_rank', rk=10, bs=10, test_size=0.05, a=0.8, b=0.1, c=0.1) :
    # if a + b + c != 1:
    #     print('(a, c, c) should be a convex combination...')
    Z = np.random.normal(0, 1, (N, r))
    X = np.random.normal(0, 1, (N, p)) + Z @ beta
    Y_x = X @ gamma 
    if noise_type == 'low_rank':
        A = np.random.randn(d, rk)
        Sigma = A @ A.T
    elif noise_type == 'full_rank':
        A = np.random.randn(d, d)
        Sigma = A @ A.T
    elif noise_type == 'diag':
        Sigma = np.identity(d)
    elif noise_type == 'block_diag':
        Sigma = generate_psd_block_diagonal_matrix(d, block_size=bs)
    else :
        print('Noise type not available!')

    Sigma = Sigma/np.linalg.norm(Sigma)
    # u, _ = np.linalg.eigh(Sigma)
    # u = np.sort(u)[::-1]
    # print(u[0])

    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 
    Y_z = Z @ A_z

    # Y_z, N_Y = Y_z/np.std(Y_z), N_Y/np.std(N_Y)
    # if np.std(Y_x) != 0:
    #     Y_x2 = (Y_x @ A_x)/np.std(Y_x @ A_x)

    # print(np.var(Y_x2), np.var(Y_z), np.var(N_Y))

    Y = a * Y_x @ A_x + b * Y_z  + c*N_Y

    return X, Y, Z, Y_x



def generate_data_Sigma(N, p, r, d, beta, gamma, A_x, A_z, Sigma, a=0.8, b=0.1, c=0.1) :
    Z = np.random.normal(0, 1, (N, r))
    X =  np.random.normal(0, 1, (N, p)) + Z @ beta 
    Y_x = X @ gamma 
    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 
    Y_z = Z @ A_z
    Y = a * Y_x @ A_x + b * Y_z  + c*N_Y

    return X, Y, Z, Y_x

def generate_data_Sigma_nonlinear(N, p, r, d, beta, gamma, A_x, A_z, Sigma, a=0.8, b=0.1, c=0.1, nl_param=2) :
    Z = np.random.normal(0, 1, (N, r))
    X = f_a(Z, beta, a=nl_param) + np.random.normal(0, 1, (N, p)) 
    Y_x = f_a(X, gamma, a=nl_param)
    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 
    Y_z = f_a(Z, A_z, a=nl_param)
    Y = a * Y_x @ A_x + b * Y_z  + c*N_Y

    return X, Y, Z, Y_x

def generate_data_mixed(N, p, r, d, beta, gamma, A_x, A_z, noise_type='low_rank', rk=10, bs=10, test_size=0.05, a=0.8, b=0.1, c=0.1) :
    # if a + b + c != 1:
    #     print('(a, c, c) should be a convex combination...')
    Z = np.random.normal(0, 1, (N, r))
    X = np.random.normal(0, 1, (N, p)) + Z @ beta
    Y_x = X @ gamma 
    if noise_type == 'low_rank':
        A = np.random.randn(d, rk)
        Sigma = A @ A.T
    elif noise_type == 'full_rank':
        A = np.random.randn(d, d)
        Sigma = A @ A.T
    elif noise_type == 'diag':
        Sigma = np.identity(d)
    elif noise_type == 'block_diag':
        Sigma = generate_psd_block_diagonal_matrix(d, block_size=bs)
    else :
        print('Noise type not available!')

    Sigma = Sigma/np.linalg.norm(Sigma)
    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 
    Y_z = Z @ A_z

    # Y_z, N_Y = Y_z/np.std(Y_z), N_Y/np.std(N_Y)
    # if np.std(Y_x) != 0:
    #     Y_x2 = (Y_x @ A_x)/np.std(Y_x @ A_x)

    # print(np.var(Y_x2), np.var(Y_z), np.var(N_Y))

    Y = f_a(Y_x @ A_x * Y_z  * N_Y, np.random.uniform(0, 1, (d, d)))

    return X, Y, Z, Y_x


def generate_mediating_data(N, p, r, d, beta, gamma, A_x, A_z, noise_type='low_rank', rk=10, bs=10, test_size=0.05, a=0.8, b=0.1, c=0.1) :
    if a + b + c != 1:
        print('(a, c, c) should be a convex combination...')
    X = np.random.normal(0, 1, (N, p))
    Z = X @ beta.T + np.random.normal(0, 1, (N, r))
    Y_x = X @ gamma 
    if noise_type == 'low_rank':
        A = np.random.randn(d, rk)
        Sigma = A @ A.T
    elif noise_type == 'full_rank':
        A = np.random.randn(d, d)
        Sigma = A @ A.T
    elif noise_type == 'diag':
        Sigma = np.identity(d)
    elif noise_type == 'block_diag':
        Sigma = generate_psd_block_diagonal_matrix(d, block_size=bs)
    else :
        print('Noise type not available!')

    # Sigma = Sigma/np.linalg.norm(Sigma)
    N_Y = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=N) 
    Y_z = Z @ A_z

    Y_z, N_Y = Y_z/np.std(Y_z), N_Y/np.std(N_Y)
    if np.std(Y_x) != 0:
        Y_x = Y_x/np.std(Y_x)

    Y = a * Y_x @ A_x + b * Y_z  + c*N_Y


    return X, Y, Z, Y_x



def compute_kl_divergence(lambda_array, dfd, dfn, bins=50):
    """
    Compute the KL divergence between the empirical distribution of `lambda_array`
    and the theoretical F-distribution with degrees of freedom `dfd` (numerator)
    and `dfn` (denominator).
    
    Parameters:
    - lambda_array: array-like, the empirical data samples
    - dfd: int, degrees of freedom for the numerator of the F-distribution
    - dfn: int, degrees of freedom for the denominator of the F-distribution
    - bins: int, optional, number of bins to use in the histogram
    
    Returns:
    - kl_divergence: float, the Kullback-Leibler divergence between the empirical
      and theoretical distribution
    """
    # Compute empirical distribution (P) using a histogram
    hist, bin_edges = np.histogram(lambda_array, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Compute theoretical distribution (Q) from F-distribution at the bin centers
    f_pdf = f.pdf(bin_centers, dfd, dfn)
    
    # Avoid division by zero by ensuring no zero entries in the theoretical distribution
    f_pdf = np.clip(f_pdf, 1e-10, None)
    
    # Compute KL divergence between the empirical (P) and theoretical (Q) distribution
    kl_divergence = entropy(hist, f_pdf)
    
    return kl_divergence