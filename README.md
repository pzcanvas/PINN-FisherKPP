# PINN-FisherKPP
 
In the past few years, there has been great interest on adopting deep neural network approaches to solve partial differential equations (PDEs). In this project we work on solving the the Fisher-KPP equation(or Fisher's equation,  Kolmogorov–Petrovsky–Piskunov equation). by data-driven deep learning methods adapted and improved from the original PINN (Physics Informed Neural Networks) method (Raissi et al, 2017), leveraging our knowledge of the long-time behavior of the solution. We also compare to other existing numerical approximation methods and assess the accuracy of our method by comparing with the exact solution, for initial and boundary conditions for which an analytic solution to Fisher-KPP exists. To the best of our knowledge this has not been attempted before for this equation.

# References

[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis,
J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Joze-
fowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mané, R. Monga, S. Moore, D. Murray,
C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke,
V. Vasudevan, F. Viégas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and
X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL
https://www.tensorflow.org/. Software available from tensorflow.org.

[2] J. Alexander and M. C. Mozer. Template-based algorithms for connectionist rule extraction.
In G. Tesauro, D. Touretzky, and T. Leen, editors, *Advances in Neural Information Processing
Systems*, volume 7. MIT Press, 1994. 

[3] J. Blechschmidt and O. G. Ernst. Three ways to solve partial differential equations with neural
networks — a review, 2021.

[4] J. Bower and D. Beeman. The Book of GENESIS: Exploring Realistic Neural Models with the
GEneral NEural SImulation System. *Biology/Neurosciences/Neural Networks*. TELOS, 1995.


[5] M. Bramson. Convergence of solutions of the Kolmogorov equation to travelling waves. *Memoirs
of the American Mathematical Society*, 44:0–0, 1983.

[6] V. Chandraker, A. Awasthi, and S. Jayaraj. A numerical treatment of fisher equation. *Pro-
cedia Engineering*, 127:1256–1262, 2015. 

[7] R. A. Fisher. The wave of advance of advantageous genes. *Annals of Eugenics*, 7(4):355–369,
1937.

[8] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau,
E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk,
M. Brett, A. Haldane, J. F. del Río, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard,
T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant. Array programming with
NumPy. *Nature*, 585(7825):357–362, Sept. 2020.

[9] S. Hasnain and M. Saqib. Numerical study of one dimensional fishers kpp equation with finite
difference schemes. *American Journal of Computational Mathematics*, 07:720–726, 2017.

[10] M. Hasselmo, E. Schnell, and E. Barkai. Dynamics of learning and recall at excitatory recurrent
synapses and cholinergic modulation in rat hippocampal region ca3. In *Journal of Neuroscience*,
1995.

[11] J. D. Hunter. Matplotlib: A 2d graphics environment. *Computing In Science & Engineering*, 9
(3):90–95, 2007.

[12] W. E. Jiequn Han, Arnulf Jentzen. Solving high-dimensional partial differential equations using
deep learning . arXiv preprint arXiv:1707.02568, 2017.

[13] K. S. Justin Sirignano. DGM: A deep learning algorithm for solving partial differential equations.
arXiv preprint arXiv:arXiv:1708.07469, 2017.

[14] I. Khalifa. Comparing numerical methods for solving the fisher equation. 2020.

[15] A. Kolmogorov, I. Petrovskii, and N. Piskunov. A study of the diffusion equation with increase
in the amount of substance, and its application to a biological problem. Vladimir M. Tikhomirov,
1991.

[16] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis. DeepXDE: A deep learning library for solving
differential equations. *SIAM Review*, 63(1):208–228, jan 2021.

[17] J. D. Murray. Mathematical biology II: Spatial models and biomedical applications, volume 3.
*Springer New York*, 2001.

[18] J. D. Murray. Mathematical biology: I. An introduction. Springer, 2002.

[19] W. Peng, J. Zhang, W. Zhou, X. Zhao, W. Yao, and X. Chen. IDRLnet: A Physics-Informed
Neural Network Library. 2021.

[20] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics informed deep learning (Part I): data-
driven solutions of nonlinear partial differential equations. *CoRR*, abs/1711.10561, 2017.

[21] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial
differential equations. *Journal of Computational physics*, 378:686–707, 2019.

[22] R. Saeed and A. Mustafa. Numerical solution of fisher–kpp equation by using reduced differen-
tial transform method. volume 1888, page 020045, 09 2017. 

[23] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski,
P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman,
N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey,  ̇I. Polat, Y. Feng,
E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero,
C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, and SciPy 1.0
Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature
Methods*, 17:261–272, 2020. 

[24] C. L. Wight and J. Zhao. Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive
physics informed neural networks. arXiv preprint arXiv:2007.04542, 2020.


