import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
import matplotlib as mpl
from matplotlib.collections import PathCollection, PolyCollection
from scipy.special import factorial2
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from enterprise import constants as const
import os
import pickle

N_psrs = '100'

cgw = 'gauss_gwb_7WN_curngwbA'

fontsize = 20
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['font.family'] = 'DejaVu Serif'

def m2(log10_rho, alpha, c):

    return (10**log10_rho)**2 * (1 + alpha*(c - 1))

def m4(log10_rho, alpha, c):

    return 3 * (10**log10_rho)**4 * (1 + alpha*(c**2 - 1))

def kurtosis(c, alpha):
    kurt = (1 + alpha * (c**2 - 1)) / (1 + alpha * (c-1))**2
    return kurt - 1

def powerlaw(Tobs, f, log10_A=-16, gamma=5):
    df = 1 / Tobs
    fyr = 1 / (365.25 * 24 * 3600)
    return ((10**log10_A) ** 2 / 12.0 / np.pi**2 * fyr ** (gamma - 3) * f ** (-gamma) * df)

def get_a_four(alpha, log10_rho, log10_c, size=1000):

    a_re = np.zeros((len(alpha), size))
    a_im = np.zeros((len(alpha), size))
    for i in range(len(alpha)):
        g1 = np.random.normal(loc=0., scale=10**log10_rho[i], size=size)
        g2 = np.random.normal(loc=0., scale=(10**log10_c[i])**0.5 * 10**log10_rho[i], size=size)
        p = np.append((1 - alpha[i]) * np.ones(size), alpha[i] * np.ones(size))
        a_re[i] = np.random.choice(np.append(g1, g2), p=p/np.sum(p), size=size)
        a_im[i] = np.random.choice(np.append(g1, g2), p=p/np.sum(p), size=size)
    return a_re, a_im

def get_psd(log10_rho, log10_c, alpha, size=1000):

    a_re, a_im = get_a_four(np.array([alpha]), np.array([log10_rho]), np.array([log10_c]))
    psd = 0.5 * np.sum(a_re**2 + a_im**2) / size
    return psd

    # return np.array([np.log10(np.mean(10**psd)) for psd in psds])

def violinplot(ax, positions, data, bandwidths, width=0.4, alpha=0.6, points=None, color='C0', label=None, xlim=None):

    added_label = False
    for pos, sample, bw in zip(positions, data, bandwidths):
        if points == None:
            kde = gaussian_kde(sample, bw_method=bw)
        else:
            kde = gaussian_kde(sample[:-points], bw_method=bw)
        if xlim == None:
            xmin = np.percentile(sample, 0.5)
            xmax = np.percentile(sample, 99.5)
        else:
            xmin = xlim[0]
            xmax = xlim[1]
        # dx = (xmax - xmin) / 100
        x_vals = np.linspace(xmin, xmax, 100)
        kde_vals = kde.evaluate(x_vals)
        kde_vals = kde_vals / kde_vals.max() * width  # scale for width of the violin
        
        if not added_label:
            ax.fill_betweenx(x_vals, pos - kde_vals, pos + kde_vals, alpha=alpha, color=color, label=label)
            added_label = True
        else:
            ax.fill_betweenx(x_vals, pos - kde_vals, pos + kde_vals, alpha=alpha, color=color)

def get_percentiles_histogram(a, nbin=50, quantiles=[16, 50, 86]):

    bins = np.linspace(np.amin(a), np.amax(a), nbin+1)
    hists = np.zeros((len(a), nbin))
    for i in range(len(a)):
        hist, edges = np.histogram(a[i], bins=bins, density=True)
        hists[i] = hist

    ps = []
    for q in quantiles:
        ps.append(np.percentile(hists, q, axis=0))

    x = edges[:-1] + 0.5 * (edges[1] - edges[0])

    return x, ps

nfs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ngs = [True]
bfs = []
sd_bfs = []
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
fig.suptitle(str(cgw), fontsize=25)

yr = 3600 * 24 * 365.25
Tobs = 10 * yr              # Observational time (10yr) in seconds 

for ng in ngs:
    psds = []
    a2s = []
    kurts = []

    for nf in nfs:

        print("nf =", nf)

        outdir = os.path.expanduser('~/Tesi-Linux/results/data/'+str(cgw)+'/ng_freq'+str(nf))
       
        chains = np.loadtxt(outdir + '/chain_1.txt')
        pars = np.loadtxt(outdir + '/pars.txt', dtype='str')

        # trova indici parametri
        log10_rho_idx = list(pars).index('crn_log10_rho')    #### CAMBIARE#############################################################################################
        #log10_rho_idx = list(pars).index('gw_hd_log10_rho')
        log10_c_idx   = list(pars).index('c')
        alpha_idx     = list(pars).index('alpha')

        log10_rho = chains[:, log10_rho_idx]
        log10_c   = chains[:, log10_c_idx]
        alpha     = chains[:, alpha_idx]
        print('len c: ', len(log10_c))
        size = 1000

        rho2 = m2(log10_rho, alpha, 10**log10_c)
        print('len rho2: ',len(rho2))
        # rho2 = 0.5 * np.sum(a_re**2 + a_im**2, axis=1) / size
        # rho4 = 0.5 * np.sum(a_re**4 + a_im**4, axis=1) / size

        rho22 = m2(log10_rho, alpha, 10**log10_c)
        # rho22 = np.sum(a_re**2, axis=1) / size
        rho4 = m4(log10_rho, alpha, 10**log10_c)
        # rho4 = np.sum(a_re**4, axis=1) / size

        '''
        mask = abs(rho2) < np.mean(rho2) + 3. * np.std(rho2)
        # TOLGO IL MASK QUA
        mask = np.ones(len(rho2), dtype='bool')
        rho2 = rho2[mask]

        mask = abs(rho22) < np.mean(rho22) + 3. * np.std(rho22)
        mask *= abs(rho4) < np.mean(rho4) + 3. * np.std(rho4)

        # TOLGO IL MASK QUA
        mask = np.ones(len(rho2), dtype='bool')

        rho22 = rho22[mask]
        rho4 = rho4[mask]
        '''
        # print(len(rho2), len(rho4))

        kurt = kurtosis(10**log10_c, alpha)
        print('len kurt: ',len(kurt))

        psds.append(np.log10(rho2))
        a2s.append(np.log10(rho4))
        #kurts.append(kurt)
        kurts.append((rho4 - 3*rho22**2) / (3*rho22**2))
        kurts[-1] = kurts[-1][abs(kurts[-1]) < 5.]
    print(np.shape(kurts))

    if ng:
        color = 'C1'
    else:
        color = 'C0'
    # plt.subplot(2, 1, 1)
    freqs = np.array(nfs) / Tobs
    width = 0.5 / Tobs

    freqs *= 1e9
    width *= 1e9

    # ax[0].violinplot(psds, freqs, showmedians=True, showextrema=False, widths=width)# , color=color)
    violinplot(ax[0], freqs, psds, [1.*np.std(psd) for psd in psds], width=width, color='tomato')
    # plt.plot(nfs, [np.median(psd) for psd in psds], ls='--', color=color)
    # plt.ylim((np.amin([np.median(psd) for psd in psds]), np.amax([np.median(psd) for psd in psds])))
    # plt.xscale('log')
    ax[0].grid(True)
    #ax[0].set_ylim((-16.3, -13.8))
    #ax[0].set_ylim((-16.3, -12))
    ax[0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].set_ylabel(r'$\log_{10} \rho^2$', fontsize=fontsize)
    # plt.subplot(2, 1, 2)
    # bw_method = [0.01, 0.01, 0.01, 0.01, 0.1]
    #print('KURTS', [np.std(kurt) for kurt in kurts])


    fracs = [0.25] * len(nfs)
    ylim = (-0.5, 3)
    ax[1].set_ylim(ylim)
    color = 'C0'
    violinplot(ax[1], freqs , kurts, [frac*np.std(kurt) for frac, kurt in zip(fracs, kurts)], width=width*0.25, color=color, label=r'$\Delta \bar{M}_4$')#, xlim=ylim)
    #ax[1].legend(loc='upper right', fontsize=13)
    # for kurt in kurts:
    #     ax[1].hist(kurt, bins=20, color=color)
    # ax[1].violinplot(kurts, freqs, showmedians=True, showextrema=False, widths=width, bw_method=0.025, points=7000) #, color=color)
    ax[1].grid(True)
    # ax[1].legend(loc='upper right')
    # ax[1].set_ylabel(r'$\Delta \bar{M}_4$', fontsize=fontsize)
    ax[1].set_ylabel(r'$\Delta \bar{M}_4$', fontsize=fontsize)
    ax[1].set_xlabel('Frequency [Hz]', fontsize=fontsize)
    #plt.subplots_adjust(hspace=0.)
    # print(len(freqs), len(bfs))

os.makedirs('./plots_v2', exist_ok=True)
os.makedirs('./plots_v2/kurt', exist_ok=True)
os.makedirs('./plots_v2/png', exist_ok=True)
os.makedirs('./plots_v2/other_stuff', exist_ok=True)

filename = str(cgw)+'_rho4_sum_'+str(N_psrs)+'psrs'

plt.savefig('./plots_v2/png/'+filename+'.png', bbox_inches='tight')
plt.savefig('./plots_v2/'+filename+'.pdf', bbox_inches='tight')

plt.cla()
plt.clf()

plt.hist(kurts[0], histtype='step', density=True)

plt.savefig('./plots_v2/'+str(cgw)+'_kurtss_'+str(N_psrs)+'psrs.png', bbox_inches='tight')

plt.savefig('./plots_v2/other_stuff/'+str(cgw)+'_fs_ng_'+str(ng)+'_rho4'+str(N_psrs)+'psrs.png', bbox_inches='tight')
plt.cla()
plt.clf()
plt.savefig('./plots_v2/other_stuff/'+str(cgw)+'_fs_crn_rho4'+str(N_psrs)+'psrs.pdf', bbox_inches='tight')
plt.cla()
plt.clf()

pl = powerlaw(Tobs, freqs[-1], -15., 13/3)
pl_med = 10**(2*np.median(log10_rho))

plt.hist(log10_rho)
plt.axvline(np.log10(pl**0.5), color='k', linewidth=5., linestyle='--')
plt.savefig('./plots_v2/other_stuff/'+str(cgw)+'_hd_stuff'+str(N_psrs)+'psrs'+'.png', bbox_inches='tight')
plt.cla()
plt.clf()

data = np.column_stack((np.mean(kurts, axis=1), np.std(kurts, axis=1)))

np.savetxt(f'./plots_v2/kurt/{cgw}_kurt_{N_psrs}psrs.txt', data)