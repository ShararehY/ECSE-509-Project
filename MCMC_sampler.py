import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import linalg as LA

def preprocess_data(data_name):
    data = pd.read_csv(data_name, sep="\t")
    datas = data.values
    n, m = data.shape
    x = np.zeros((97, 8))
    y = np.zeros((97, 1))
    for i in range(n):
        x[i] = datas[i][1:9]
        y[i] = datas[i][9]
    scaler = preprocessing.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    scaler = preprocessing.StandardScaler().fit(y)
    y_scaled = scaler.transform(y)
    return x_scaled,y_scaled

def plot_xy(x_scaled,y_scaled):
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        for j in range(2):
            axs[i, j].scatter(x_scaled.transpose()[i*2+j], y_scaled, alpha=0.5)
            # axs[0, 0].set_xlabel('x1')
            axs[i, j].set_ylabel('y')
            axs[i, j].set_title('x'+str(i*2+j+1))
    plt.show()

def estimate_prior(x_scaled,y_scaled):
    w = np.matmul(LA.inv(np.matmul(x_scaled.transpose(), x_scaled)), np.matmul(x_scaled.transpose(), y_scaled))
    res=y_scaled-np.matmul(x_scaled,w)
    sigma2=sum(res**2)/len(y_scaled)
    wlambda=np.trace(LA.inv(np.matmul(x_scaled.transpose(),x_scaled))*sigma2)/8
    return wlambda,sum(res**2)/len(y_scaled)

def MAP():
    return np.matmul(LA.inv(np.matmul(x_scaled.transpose(),x_scaled)+sigma2/wlambda*np.eye(8)),np.matmul(x_scaled.transpose(),y_scaled))

def log_prob(w):
    return -0.5*np.matmul(np.matmul((w-mu).transpose(),LA.inv(var_cov)),(w-mu))

def log_prob_1D(wi,i):
    return -0.5/var_cov[i][i] * (wi - mu[i])*(wi - mu[i])

def proposal(x, stepsize):
    return np.random.uniform(low=x - 0.5 * stepsize,
                             high=x + 0.5 * stepsize,
                             size=x.shape)

def p_acc_MH(w_new, w_old, log_prob):
    return min(1, np.exp(log_prob(w_new) - log_prob(w_old)))

def sample_MH(w_old, log_prob, stepsize):
    w_new = proposal(w_old, stepsize)
    accept = np.random.random() < p_acc_MH(w_new, w_old, log_prob)
    if accept:
        return accept, w_new
    else:
        return accept, w_old

def build_MH_chain(init, stepsize, n_total, log_prob):
    n_accepted = 0
    chain = [init]
    for _ in range(n_total):
        accept, state = sample_MH(chain[-1], log_prob, stepsize)
        chain.append(state)
        n_accepted += accept
    acceptance_rate = n_accepted / float(n_total)
    return chain, acceptance_rate

def plot_samples(chainx, n, log_prob, ax, orientation='vertical', normalize=True,
                 xlims=(-1, 1), legend=True):
    from scipy.integrate import quad
    chain=[]
    #n=6
    for i, item in enumerate(chainx):
        chain.append(item[n][0])
    ax.hist(chain, bins=50, density=True, label="MCMC samples",
           orientation=orientation)
    # numerically calculate the normalization constant of PDF
    if normalize:
        Z, _ = quad(lambda x: np.exp(log_prob_1D(x,n)), -np.inf, np.inf)
    else:
        Z = 1.0
    xses = np.linspace(xlims[0], xlims[1], 1000)
    yses = [np.exp(log_prob_1D(x,n)) / Z for x in xses]
    if orientation == 'horizontal':
        (yses, xses) = (xses, yses)
    ax.plot(xses, yses, label="true distribution")
    ax.set_title('w('+str(n+1)+') distribution')
    if legend:
        ax.legend(frameon=False)
    plt.show()

def MAP_sampled(chain):
    w_mean=[]
    for n in range(8):
        mean=0
        for item in chain:
            mean+=item[n][0]
        w_mean.append(mean/len(chain))
    return w_mean

def visualize(chain):
    fig, axs = plt.subplots(4, 2)
    for i in range(4):
        for j in range(2):
            n=i*2+j
            sample_mean = []
            step = []
            mean = 0
            for count, item in enumerate(chain):
                mean += item[n][0]
                if count % 100 == 0 and count != 0:
                    sample_mean.append(mean / 100)
                    step.append(count)
                    mean = 0
            axs[i, j].plot(step,sample_mean)
            axs[i, j].set_ylabel('mean'+str(i * 2 + j + 1))
    plt.show()
    return sample_mean

if __name__ == "__main__":
    x_scaled,y_scaled=preprocess_data('prostate.data')
    plot_xy(x_scaled,y_scaled)
    wlambda,sigma2=estimate_prior(x_scaled,y_scaled)

    var_cov=LA.inv(1/sigma2*np.matmul(x_scaled.transpose(),x_scaled)+1/wlambda*np.eye(8))
    mu=np.matmul(var_cov,1/sigma2*np.matmul(x_scaled.transpose(),y_scaled))

    chain, acceptance_rate = build_MH_chain(np.ones((8,1)), 0.1, 80000, log_prob)
    print("Acceptance rate: ",acceptance_rate)
    fig, ax = plt.subplots()
    plot_samples(chain[4000:],0, log_prob, ax)
    #vis=visualize(chain[0:10000])
    print("estimated MAP:" ,np.array(MAP_sampled(chain[4000:])))
    print("theoretical MAP:" ,np.array(MAP()[:,0]))

    x_cov=np.cov(x_scaled.T)
    eigg,v_eig=LA.eig(x_cov)
    print('total variance percentage=',(eigg[0]+eigg[1])/sum(eigg)*100)
    p_comp=np.array((v_eig[0], v_eig[1]))    #didn't need sorting, first two were the biggest ones
    x_transformed=x_scaled.dot(p_comp.transpose())
    plt.scatter(x_transformed[:,0], x_transformed[:,1])
    plt.title('projected x in 2 dimension')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.show()