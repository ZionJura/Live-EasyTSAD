# Deep Isolation Forest for Anomaly Detection

<!-- page 1 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
1

Deep Isolation Forest for Anomaly Detection

Hongzuo Xu , Guansong Pang , Yijie Wang
and Yongjun Wang

Abstract—Isolation forest (iForest) has been emerging as arguably the most popular anomaly detector in recent years due to its general effectiveness across different benchmarks and strong scalability. Nevertheless, its linear axis-parallel isolation method often leads to (i) failure in detecting hard anomalies that are difficult to isolate in high-dimensional/non-linear-separable data space, and (ii) notorious algorithmic bias that assigns unexpectedly lower anomaly scores to artefact regions. These issues contribute to high false negative errors. Several iForest extensions are introduced, but they essentially still employ shallow, linear data partition, restricting their power in isolating true anomalies. Therefore, this paper proposes deep isolation forest. We introduce a new representation scheme that utilises casually initialised neural networks to map original data into random representation ensembles, where random axis-parallel cuts are subsequently applied to perform the data partition. This representation scheme facilitates high freedom of the partition in the original data space (equivalent to non-linear partition on subspaces of varying sizes), encouraging a unique synergy between random representations and random partition-based isolation. Extensive experiments show that our model achieves significant improvement over state-of-the-art isolation-based methods and deep detectors on tabular, graph and time series datasets; our model also inherits desired scalability from iForest.

Index Terms—Anomaly Detection, Isolation Forest, Deep Representation, Ensemble Learning.

✦

1
INTRODUCTION
A

NOMALY detection has broad applications in various domains, such as detection of insurance fraud and financial crime, surveillance of complex systems like data centres and spacecraft, and identification of attacks and potential threats in cyberspace. Given these important applications, this task has been a popular research topic for decades, and numerous anomaly detection approaches have been introduced [1], [2].

In recent years, isolation forest (iForest) [3] has been emerging as arguably the most popular anomaly detector due to its general effectiveness across different benchmarks and strong scalability. Compared to many existing methods such as distance/density-based methods, iForest better captures the key essence of anomalies, i.e., “few and different”. It does not introduce extra assumptions of data characteristics, thus showing consistently effective performance across diverse datasets. Also, iForest is with linear time complexity, which is often a very appealing advantage in many industrial applications when there are large-scale data and strict requirements of time efficiency. Concretely, iForest uses an ensemble of isolation trees (iTree), in which each iTree is grown by iteratively branching. Leaf nodes are built by using random cuts in the values of randomly selected features until the data objects are isolated. Data abnormality is estimated according to the average depth traversing from the root node to the isolated leaf node in these iTrees.

Nevertheless, an explicit major issue is that it cannot handle hard anomalies (e.g., the anomalies that can be

• Hongzuo Xu, Yijie Wang and Yongjun Wang are with the College of Computer, National University of Defense Technology, Changsha 410073, PR China. Hongzuo Xu and Yijie Wang are also with the Science and Technology on Parallel and Distributed Processing Laboratory. E-mail: {xuhongzuo13, wangyijie, wangyongjun}@ nudt.edu.cn • Guansong Pang is with the School of Computing and Information Systems, Singapore Management University Singapore, 178902, Singapore. E-mail: gspang@smu.edu.sg • Guansong Pang and Yijie Wang are corresponding authors.

only isolated in higher-order subspaces by looking into the combination of multiple features) because it treats all features separately and considers only one feature per isolation operation. Fig. 1 exemplifies this issue with a simple toy example. Anomalies (represented as red triangles) are surrounded by ring-shaped normal samples, which cannot be isolated by either x-axis or y-axis slicing cuts. Although these anomalies might be finally isolated by multiple cuts, it results in indistinguishable isolation depth in iTrees compared to normal data. The failure of recalling these anomalies induces high false negative errors. Given that anomalies often contain critical information related to potential accidents or faults, those false negatives may cause serious consequences. Therefore, this issue has been a major bottleneck hindering the performance of iForest, particularly on datasets with high-dimensional/non-linearseparable data spaces.

Another inherent imperfection of iForest is that it assigns unexpectedly low anomaly scores to artefacts introduced by the algorithm itself, which is revealed as the “ghost region” problem in [4]. To clearly demonstrate this issue, we visualise the data distribution of three 2-D synthetic datasets used in [4] and anomaly score maps generated by iForest in the first two columns in Fig. 2, respectively. As can be seen from all three anomaly score maps, iForest assigns clearly lower anomaly scores to some artefact regions, i.e., the four rectangular areas centred around the presented data objects in the single-blob dataset (1st row), the upper right and bottom left clique areas in the two-blob dataset (2nd row), and the vertical rectangular areas along the sinusoid in the sinusoidal dataset (3rd row). However, these artefact regions are similar to other regions that contain no samples or have similar radial distances to the presented data objects. The unusually lower anomaly scores are due to the intrinsic algorithmic bias of iForest, i.e., only axis-parallel partitions are admitted in iTree construction. Instead, an expected anomaly score map should smoothly approximate circular

arXiv:2206.06602v4 [cs.LG] 9 Jun 2023

<!-- page 2 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
2

Projected Space
Projected Space
Projected Space
Projected Space

Fig. 1. Illustration of the hard anomaly challenge. The left figure is a synthetic dataset, where red triangles are anomalies and blue points are normal samples. The following figures are projected spaces (i.e., random representations) created by our method. Linear data partition (in either vertical/horizontal or oblique forms) used in existing methods cannot effectively isolate these hard anomalies in the original data space. By contrast, these anomalies are possible to be exposed in newly created spaces.

iForest
PID
EIF
DIF (ours)

Fig. 2. Illustration of the algorithmic bias problem. The three figures on the far left are three synthetic datasets, and the following panels of each dataset are the anomaly score maps produced by iForest and its two state-of-the-art extensions – PID and EIF – and our method DIF. Score maps indicate anomaly score distribution in the full data space (deeper colour indicates higher abnormality). Black contour lines in each score map denote the 99th percentile of the anomaly scores of the original data, which can be seen as the boundary of data normality predicted by each anomaly detector. Red triangles represent possible anomalies. There are some artefact regions in the score maps of iForest and its two extensions, and these methods may fail to identify the anomalies since these anomalies are assigned similarly low anomaly scores as the presented normal samples (they are included in the predicted normal areas). By contrast, DIF produces more accurate and smooth score maps, and DIF can successfully assign a high abnormal degree to anomalies compared to those original data samples.

contour lines w.r.t. the density of the presented data. This problem also leads to false negative errors, i.e., iForest may fail to detect possible anomalies in these “ghost regions” such as the red triangles in Fig. 2 as they are assigned similarly low anomaly scores as some of the presented normal objects and included in the normal areas.

There have been a number of extensions of iForest introduced over the years [4], [5], [6], [7], [8], [9]. These extensions attempt to devise more advanced isolation methods by (i) using novel hierarchical criteria to selectively pick splitting thresholds and/or dimensions [6], [8], [9] or (ii) using optimal/random hyper-planes [4], [5], [7]. These improvements successfully contribute to better detection performance, but their isolation methods are still confined by linear partition. Instead, data partition in the isolation process is expected to form non-linearly arbitrary shapes in the data space to handle hard anomalies in complicated datasets and eliminate the algorithmic bias in derived anomaly scores. We show two state-of-the-art extensions (PID [6] and EIF [4]) in Fig. 2. PID [6] selectively chooses splitting dimensions

and uses similar axis-parallel isolation operations. Thus, its score maps also encounter the same algorithmic bias problem. EIF [4] works better due to the use of hyper-planes with random slopes and intercepts as branching criteria, eliminating the limits of axis-parallel partitions to some extent. However, the partitions still operate linearly, which works ineffectively on challenging datasets where only nonlinear partitions are useful in isolating anomalies from the other data. This problem is reflected by the rather wrinkled anomaly score contours in all three datasets in Fig. 2.

Therefore, the key to these limitations becomes very clear now, i.e., a step further is required to liberate the isolation method from the underlying linear constraints. To this end, this paper proposes a novel extension of iForest named DEEP ISOLATION FOREST (DIF for short). The key idea in DIF is to harness the strong representation power of neural networks to map the original data into a group of new data spaces, and non-linear isolation can be easily achieved by performing simple axis-parallel partitions upon these newly created data spaces (equivalent to nonlinear partition on subspaces of varying sizes in the original data space). Specifically, we propose a novel representation scheme – random representation ensemble – produced by optimisation-free deep neural networks. These networks are only casually initialised and do not involve any optimisation or training process. DIF then simply utilises random axisparallel cuts upon these representations. The randomness in our representation scheme allows high freedom of the partition in the original data space, encouraging a unique synergy effect between random representations and random partition-based isolation. This facilitates effective isolation of hard anomalies and eliminates the algorithmic bias, thus significantly enhancing the detection performance. Fig. 1 shows four random representations created by DIF, where those hard anomalies are possible to be exposed and easily isolated by using a few axis-parallel cuts. Besides, as shown in Fig. 2, DIF accurately assigns anomaly scores to the presented data objects and has a smooth estimation of the anomaly scores in the other regions, alleviating the aforementioned algorithmic bias problem.

Further, in our specific implementation of DIF, we propose the Computation-Efficient Representation Ensemble method (CERE) and the Deviation-Enhanced Anomaly Scoring function (DEAS). With the help of CERE, DIF can efficiently produce the representation ensemble by taking full advantage of parallel accelerators in the mini-batch calculation, largely eliminating the computational overhead. DEAS leverages the hidden quantitative information enclosed in the mapped dense representations along with qualitative comparisons. This additional information enables a more accurate assessment of the isolation difficulty of data objects, leading to a better anomaly scoring function.

Our main contributions are summarised as follows:

• We
propose
the
DEEP ISOLATION FOREST
(DIF)
method. A new isolation method is introduced, which
enables non-linear partition on subspaces of vary-
ing sizes, offering a more effective anomaly iso-
lation solution than the current methods that are
capable of linear isolation only. The resulting ap-
proach can better handle hard anomalies in complex
datasets with high-dimensional/non-linear-separable

<!-- page 3 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
3

data spaces and eliminate the algorithmic bias. We also show that DIF is a high-level generalisation of iForest and its very recent extension EIF.

• We propose a novel representation scheme, the random representation ensemble, in which only casually initialised neural networks are required. This representation scheme facilitates high freedom of partitions in the original data space. The unique synergy between random representations and random partition-based isolation brings excellent randomness and desired diversity into the overall ensemble-based abnormality estimation process.

• We propose the representation ensemble method CERE to ensure that DIF can inherit good scalability from iForest. The anomaly scoring function DEAS is introduced to leverage additional quantitative information enclosed in mapped dense representations, enhancing the quality of anomaly scoring of DIF.

• DIF offers a data-type-agnostic anomaly detection solution. We show that DIF is versatile to detect anomalies in different types of data by simply plugging in corresponding randomly initialised neural networks in the feature mapping.

Extensive experiments on a large collection of real-world datasets, including not only tabular data but graph and time series data, show that: (i) DIF significantly outperforms iForest and its state-of-the-art extensions (Sec. 6.2); (ii) DIF also achieves remarkable improvement compared to the ensemble of advanced deep anomaly detectors (Sec. 6.2); (iii) DIF has desired scalability on high-dimensional, largescale datasets (Sec. 6.3) and presents good robustness to anomaly contamination (Sec. 6.4); (iv) the significance of the synergy between random representations and random partition-based isolation is justified by comparing to several alternatives (Sec. 6.5); and (v) the contribution of CERE and DEAS is separately verified in our ablation study (Sec. 6.6).

2
RELATED WORK

2.1
Anomaly Detection

Anomaly detection has been intensively studied in the last decades by using different data characteristics like distance, density, clustering membership, or probability [1]. Recent studies [10], [11], [12], [13] devise deep anomaly detection models based on the representation learning, while some other methods use concepts like reconstruction [14] or deviation [15] in the deep learning framework as scoring function. Surveys and comparative studies can be found in [2], [16], [17]. Also, some practical anomaly detection tools, e.g., [18], [19], [20], [21], are developed to facilitate the use of anomaly detection models in real-world applications.

2.2
Isolation Forest and Its Extensions

iForest [3] is a very popular anomaly detection method that identifies anomalies according to the isolating difficulty in the data space. Although iForest has shown effective performance on several benchmarks, a number of its extensions have been proposed, attempting to fix its notorious vulnerabilities and obtain better detection performance.

The mainstream of these extensions focuses on devising more effective isolation methods. SCIF [5] introduces a non-axis-parallel way as branching criteria, i.e., instead of selecting one feature during splitting, an optimal slicing hyper-plane is used. EIF [4] also uses hyper-planes but with random slopes and intercepts. Another work [7] further fixes the empty branching problem of EIF by choosing splitting thresholds from the range of projected values onto the slope direction. PID [6] selectively picks dimensions that have greater variance and selects split points according to the sparsity of the partition branches. The literature [9] proposes a probability-based method to find better split values than random splitting. These extensions normally achieve better performance by introducing non-axis-parallel and/or heuristic partition. However, the major problem is that they still rely on linear isolation operations, which means it is also hard to handle complicated data that require nonlinear partitions as the isolation method. Also, they suffer from the aforementioned artefact problem due to the implicit algorithmic bias hidden in the isolation process. There are also some extensions employing the nearest neighbour information into isolation process, e.g., LeSiNN [22] and iNNE [23]. The literature [24] further uses locally-sensitive hashing to extend the isolation mechanism to any distance measures and data types. These distance-based methods introduce an extra assumption, i.e., anomalies are far from other data objects. However, this assumption does not always hold since clustered anomalies are also very close to their adjacent neighbours [5]. The detection performance is also sensitive to the choice of distance metrics.

Another angle is to enhance the scoring method. The literature [25] introduces a path-weighted scoring method and a probability-based aggregation function. PID [6] redefines the scoring function according to the sparsity rather than depth in the tree. Similarly, these extensions are still vulnerable due to their linear isolation methods.

2.3
Deep Ensembles

Deep ensemble [26], a simple framework that combines prediction results of a group of independently trained networks together, has garnered much interest. It can improve prediction accuracy and provide uncertainty estimation in a simple framework without modifying the original working pipeline. Similarly to other ensemble-based approaches, the quality of deep ensembles also largely hinges on the diversity of its members. Besides, albeit simple, deep ensembles still induce considerably larger computational costs. Thus, many related studies attempt to address these two key limitations, e.g., [27] uses repulsive terms to ensure individual diversity, [28] increases diversity by adversarially preventing features from being conditionally predictable from each other, and [29] proposes a distilled model that can absorb as much function diversity inside the ensemble as possible. In addition, this framework inspires related studies on anomaly detection. The literature [30], [31] combines a group of autoencoders and uses the median of reconstruction errors as anomaly scores. To improve member diversity, these autoencoders are sparsely connected.

Our work also involves the process of integrating neural networks. From the deep ensemble aspect, our work can

<!-- page 4 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
4

effectively tackle the above two key issues. The diversity between ensemble members can be guaranteed and the calculation efficiency can be well maintained since only initialised networks are required. Our work may also provides valuable insights into the deep ensemble research line.

3
PROBLEM STATEMENT AND NOTATIONS

Let D = {o1, · · · , oN} be a dataset with N data objects, anomaly detection is to give a scoring function f : D 7→RN

that estimates the abnormality of each data object. Different from many existing studies that only focus on an individual data type, we do not restrict the type of data objects in this work, which means they can be multi-dimensional vectors, time series data, or graphs. Throughout the paper, we use calligraphic fonts for sets, script typeface for functions, bold lowercase letters for vectors, bold uppercase letters to denote matrices. Table 1 summarises main notations.

TABLE 1
The main notations used in the paper

Format
Notations – Descriptions

Calligraphic fonts
D – datasets, X – representations,
T – the set of iTrees, P – data pools in iTree nodes

Script typeface
F – anomaly scoring function,
G – representation function

Bold lowercase letters
o – original data objects, x – representation vectors,
p, q – random vectors used in CERE

Bold uppercase letters
W – weight matrices of neural networks,
P, Q – matrices used in CERE

Others

r – number of representations,
t – number of iTrees per representation,
T – total number of iTrees,
p(·|·) – traversed node path,
g(·|·) – averaged deviation degree

4
PRELIMINARIES: ISOLATION FOREST

For the sake of clarity, we recall the basic procedure of isolation forest (iForest) [3]. A basic structure named isolation tree (iTree for short) is proposed. iTree τ is essentially a binary tree, and each node in the tree corresponds to a pool of data objects. A subset containing n data objects is used as the data pool of the root node, which is randomly subsampled from the whole dataset. iTree τ grows by recursively isolating data objects in the leaf node (i.e., a disjoint partition of data objects into two child nodes) in a topdown fashion until remaining one data object in the node or reaching the maximum depth limit. iForest uses a simple isolation method that performs a comparison between the j-th dimension of the data object o(j) and a splitting value η as the branching criterion of each data object o, where j and η respectively denote a randomly selected feature index and a split value within the range of available values of the j-th feature. Each data object o has a traversing path p(o|τ) in the iTree τ, and length of the traversed path, |p(o|τ)|, can be naturally viewed as an indication of the abnormal degree of o (anomalies are often easier to be isolated with shorter path length). iForest constructs a forest of T iTrees T ={τi}T

i=1. The anomaly score of data object o is calculated based on its averaged path length Eτi∈T (|p(o|τi)|) over all of the iTrees

in the forest T , i.e., FiFoerst(o|T ) = 2−Eτi∈T

|p(o|τi)|

C(T ) , where
C(T) is a normalising factor.

iForest uses a linear axis-parallel isolation method that only considers one dimension each time, and existing extension work introduces hyper-plane-based isolation that involves multiple dimensions, but similarly, only linear partition is admitted. These current isolation methods are limited to effectively handle hard anomalies that cannot be isolated using linear partitions on individual features or simple combinations of multiple features. Additionally, these existing methods generally suffer from the algorithmic bias brought by constraints hidden in their isolation strategies.

5
DEEP ISOLATION FOREST

We require a new isolation method that is unleashed from the above constraints so that it can effectively isolate those hard anomalies and avoid algorithmic bias. To this end, we introduce the DEEP ISOLATION FOREST (DIF) method. In a nutshell, DIF constructs an ensemble of representations derived from deep neural networks, and simple axis-parallel isolation is operated upon new data spaces to build iTrees in the forest. Instead of following the deep ensemble framework that combines independently trained neural networks, we use an ensemble of random representations produced by optimisation-free neural networks that only require simple casual initialisation. This new representation scheme allows high freedom of the partition in the original data space. Each feature in the newly projected representation space is based on non-linear interactions among a number of original features, meaning that axis-parallel partitions of new space can form effective non-linear cuts in the original space. This way successfully liberates the slicing cuts from current linear constraints. Those hard anomalies that cannot be easily isolated in the original data space are possible to be exposed in representation spaces, and they can be isolated via fewer cuts and lead to clearer abnormality. Meanwhile, a unique synergy between random representations and random partition-based isolation can facilitate the overall ensemble-based abnormality estimation.

5.1
Formulation of DIF

DIF first produces the random representation ensemble via optimisation-free neural networks, which is defined as

G (D) =

Xu ⊂Rd
Xu = ϕu(D; θu)

r

u=1,
(1)

where r is the ensemble size, ϕu : D7→Rd is the network that maps original data into new d-dimensional spaces, and the network weights in θu are randomly initialised. Each representation is assigned with t iTrees, and a forest T ={τi}T

i=1 containing T=r×t iTrees is constructed. iTree τi of X is initialised by a root node with a set of projected data P1 ⊂X . The k-th node with the data pool Pk is branched into two leaf nodes with disjoint subsets, i.e., P2k = {x|x(jk) ≤ ηk, x ∈Pk} and P2k+1 = {x|x(jk) > ηk, x ∈Pk}, where jk is selected uniformly at random among all the dimensions of the newly created data space {1, · · · , d}, x(jk) is the jkth dimension of the projected data object, and ηk is a split value within the range of {x(jk)|x ∈Pk}.

After constructing T , the abnormality of a data object o is evaluated by the isolation difficulty in each iTree of the forest T . The scoring function is defined as

F(o|T ) = Ωτi∼T I(o|τi),
(2)

<!-- page 5 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
5

where I(o|τi) denotes a function to measure the isolation difficulty in iTree τi, and Ωdenotes an integration function.

5.2
Implementation of DIF

There are two main components in DIF, i.e., random representation ensemble function G and isolation-based anomaly scoring function F. To improve the time efficiency of representation function G , we propose Computation-Efficient deep Representation Ensemble method (CERE), in which all the ensemble members can be computed simultaneously in a given mini-batch. To further improve the accuracy of anomaly scoring, we propose Deviation-Enhanced Anomaly Scoring function (DEAS) by leveraging the hidden quantitative information enclosed in the projected dense representations along with qualitative comparisons.

5.2.1 CERE: Computation-efficient Deep Representation Ensemble Method Successively feeding raw data into r independent networks in Eq. (1) can induce a considerably high memory and time overhead. To inherit outstanding scalability of the original iForest, we introduce CERE to efficiently implement the representation ensemble function G (D).

Let W ∈Rm×n be a weight matrix of a neural network layer, then following [32], we use a tuple of small random vectors pi ∈Rm and qi ∈Rn to yield a rank-one matrix via multiplication, which is used to derive the full weight matrices of each ensemble member. Formally, based on a base weight matrix W0, the weight matrix Wi of the i-th ensemble member is generated as

Wi = W0 ◦(piq⊤

i ),
(3)

where ◦denotes the Hadamard product.

The mapping process of incoming neurons x ∈Rm and the weight Wi can be further derived as follows:

W⊤

i x = (W0 ◦piq⊤

i )⊤x

= W⊤

0 (x ◦pi) ◦qi.
(4)

Given r tuples of weight vectors {⟨pi, qi⟩}r

i=1 and
a mini-batch of data X
∈Rb×m with mini-batch size
b and dimension m, the ensemble of mapped results
{XW1, · · · , XWr} can be calculated via





XW1
XW2

· · ·
XWr



=







X
X
· · ·
X



◦





P1
P2
· · ·
Pr







W0

!

◦





Q1
Q2

· · ·
Qr



,
(5)

where each row in Pi and Qi is duplicated pi and qi. Let 1b be a all-one vector with size b, Pi and Qi are obtained via Pi = 1bp⊤

i and Qi = 1bq⊤

i . The above vectorisation derivation allows the deep representation ensemble process in DIF to be efficiently calculated. With the help of CERE, the time complexity of the ensemble process is similar to the feed-forward process of a single neural network since all the ensemble members can be computed simultaneously in a given mini-batch. Eq. (5) requires additional Hadamard product steps, but this operation is very cheap compared to matrix multiplication. One possible limitation is the batch size in Eq. (5). A minibatch of r × t objects is simultaneously computed, which is

larger than conventional settings. However, as computation within a mini-batch is automatically parallelisable, increasing the batch size incurs almost no time overhead. As for memory cost, this process is also feasible in a typical device because DIF does not involve optimisation, i.e., gradients are not calculated and saved. For example, a dataset with 10,000 features costs about 3GB of memory when using the recommended ensemble size r=50 and the batch size b=64.

Let Φ be the L-layer neural network using the newlydefined feed-forward step in Eq. (5). The ensemble of representations can be directly generated as

GCERE(D) = Φ

 D; Θ

 =

Xi ⊂Rd r

i=1,
(6)

where Θ=

Wl, {p(l,i)}r

i=1, {q(l,i)}r

i=1

L

l=1. Note that other operations like activation or pooling and some layers that do not use a weight matrix are processed sequentially.

5.2.2 DEAS: Deviation-enhanced Anomaly Scoring Function We further introduce a new anomaly scoring function DEAS. Recall that the standard anomaly scoring process in iForest only uses the length of the traversed path, i.e., all nodes are considered to have the same importance. The path length only provides limited information, which may not sufficiently delineate the isolation difficulty of data objects. Except for qualitative comparison in each node, additional quantitative information is readily available to be leveraged, such as relation between the feature values of data objects and the branching threshold.

Motivated by this, we utilise the deviation degree of the feature value to the branching threshold as additional weighting information to further improve the measurement of isolation difficulty. These deviation degrees are important indicators to the isolation difficulty because the feature values in newly created data spaces are typically densely distributed and these deviations reflect the local density in the projected space., e.g., a small deviation value indicates that the slicing cut is on a dense region and thus is hard to isolate the data object. Specifically, let xu be the corresponding representation of a data object o in an iTree τi. p(xu|τi) = {1, · · · , K} is its traversed node path. We define the averaged deviation degree of xu in τi as

g(xu|τi) =
1
|p(xu|τi)|

X

k∈p(xu|τi)

|x(jk)

u
−ηk|.
(7)

We further combine the path length |p(xu|τi)| as in iForest and the deviation measure in Eq. (7) to specify the function in Eq. (2) by defining our deviation-enhanced isolation anomaly scoring function as:

FDEAS(o|T ) = 2−Eτi∈T

|p(xu|τi)|

C(T )
× Eτi∈T

 g(xu|τi)

,
(8)

where the first term is the averaged depth used as anomaly scores in iForest and the second term is the deviation-based anomaly score we introduce.

5.3
Algorithm of DIF

Algorithm 1 presents the procedure of the construction of deep isolation trees T . Step (2) prepares r random representations, and t isolation trees {τi}t

i=1 are built upon each

<!-- page 6 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
6

representation in Steps (4-15). For each isolation tree τi, a subset of transformed data objects P1 is first randomly subsampled in Step (5) to initialise the root node. Each leaf node Pk is then iteratively split by using a comparison branching criteria based on a randomly selected representation dimension jk and a split point ηk in Steps (6-13).

We report the anomaly scoring procedure in Algorithm
2. Data object o is transformed to vectorised representations
{x}r

u=1 in Step (1) After the initialisation in Step (4), the data object traverses each tree τi by the criteria of each node and reaches the final node, during which the traverse path p(xu|τi) and the accumulated difference β are recorded in Steps (5-12). The path length |p(xu|τi)| and the deviation g(xu|τi) in iTree τi are calculated in Step (13). The anomaly score of o is calculated and returned in Steps (16-17).

Algorithm 1 Construction of Deep Isolation Trees
Input: D - input dataset
Output: T - forest of deep isolation trees

1: Initialise T ←∅
2: Generate representations {Xu}r
u=1 via GCERE
3: for u = 1 to r do
4:
for i = 1 to t do
5:
Initialise an isolation tree τi by setting the root node
using P1 ⊆Xu, |P1| = n
6:
while Pk is a leaf node of tree τi do
7:
if |Pk| > 1 and the depth is smaller than J then
8:
Randomly select a dimension jk ∈{1, · · · , d}
9:
Randomly select a split point ηk between the
max and min values of dimension jk in Pk
10:
P2k ←{x|x(jk) ≤ηk, x ∈Pk}
11:
P2k+1 ←{x|x(jk) > ηk, x ∈Pk}
12:
end if
13:
end while
14:
T ←T ∪τi
15:
end for
16: end for
17: return T

Algorithm 2 Deviation-enhanced Anomaly Scoring Input: o - data object, T - set of deep isolation trees Output: anomaly score FDEAS(o|T )

1: Generate representations {xu}r
u=i via GCERE
2: for u = 1 to r do
3:
for i = 1 to t do
4:
Initialise k ←1, β ←0, p(xu|τi) ←∅
5:
while |Pk| > 1 and not reaching J do

6:
if x(jk)

u
≤ηk then
7:
k ←2k
8:
else
9:
k ←2k + 1
10:
end if
11:
p(xu|τi) ←p(xu|τi) ∪k, β ←β + |x(jk)

u
−ηk|
12:
end while
13:
g(xu|τi) ←β/|p(xu|τi)|
14:
end for
15: end for

16: return FDEAS(o|T )←2−Eτi∈T
|p(xu|τi)|

C(T )
×Eτ∈T

 g(xu|τi)



5.4
Theoretical Analysis

5.4.1
Time Complexity Analysis

We first analyse the time complexity for the production of the random representation ensemble via CERE (i.e., Step 2 in Algorithm 1). Let the input data D be a tabular dataset with size N ×D. Multi-layer perceptron network is used for ϕ. The used network ϕ comprises L layers, and the l-th layer is with dl hidden units, the representation dimension is d. We use CERE to implement the ensemble with r members within each mini-batch, and thus the whole feed-forward computation induces O(r×N ×(Dd1+dld+PL−1

l=1 dldl+1)). Only feed-forward steps are required in DIF, and the number of hidden units and representation dimension is generally small. Thus, this process is linear w.r.t. both data size and dimensionality, which does not introduce much extra computational overhead than the original iForest. In terms of the subsequent process of the iTree construction, given the depth limit J, we have the maximum 2J−1

splits (Steps (8-11)) during the growth of each iTree. For a node with n samples, each split takes O(n) complexity in determining the maximum and minimum value of the selected dimension and the assigning process. The overall process induces O(2J−1 ×n×r ×t). J and n often use fixed small values (typically 8 and 256 respectively). Therefore, the overall complexity is linear w.r.t. the ensemble size r ×t. As for the process in Algorithm 2, the traversing process takes a similar computation process, which has linear time complexity w.r.t. the size of testing sets and the ensemble size. Overall, according to the above analysis, the time complexity of DIF is O(ND(r × t)). It has linear complexity w.r.t. data size, dimensionality, and ensemble size, which inherits desired scalability from iForest.

5.4.2
DIF as a Generalisation of iForest and EIF

EIF [4] is a recent extension of iForest [3], which has been shown to be a generalisation of iForest in [4]. We show that DIF can be viewed as a further higher-level generalisation of isolation methods used in both iForest and EIF, i.e., the branching criteria used in iForest and EIF can be transformed into the format of DIF.

Let o ∈RD be a vectorised data object. Recall that the branching criterion in DIF is ϕ(o)(j) ≤η. Both iForest and EIF are special cases of DIF when the neural network ϕ is with one linear layer parameterised by a weight matrix W, i.e., ϕ(o) = W⊤o. iForest splits the node by using the criterion o(j) ≤η, while DIF degrades to iForest if the weight matrix is set as an identity matrix, i.e., W = ID. EIF uses a slicing hyper-plane for each node branching, and the slope of its hyper-plane is a normal vector k ∈RD, and k(i) ∼N(0, 1), ∀i ∈{1, · · · D}. The intercept of the hyper-plane p ∈RD is uniformly selected over the range of possible values at each branching point. The branching criterion is (o−p)·k ≤0, which is equivalent to o·k ≤p·k. We can fulfil exactly the same operation in DIF when the weight matrix satisfies W ∈RD×1. The elements in W should also be initialised by a normal distribution N(0, 1) to satisfy W = k. Additionally, the splitting point η = p · k can be understood as a standard random vector p sampled in possible values with a Gaussian noise k.

<!-- page 7 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
7

5.5
Discussions

The power of DIF mainly depends on (i) the strong representation ability of neural networks, (ii) the discard of optimised representations, and (iii) the synergy between random representations and random partition-based isolation, which are respectively discussed below.

5.5.1 Representation Ability of Neural Networks Neural networks have strong representation power, even for randomly initialised networks. As shown in Fig. 3 where the new data spaces are generated by random neural networks, the randomness in these initialised networks can create highly diversified data spaces, on which simple axisparallel cuts can be equivalent to sophisticated slicing cuts in the original data space. Non-linear activation functions can effectively tweak and fold partition bounds to embed non-linearity into the isolation process, even though the networks are not optimised at all. On the other hand, there have been different deep learning architectures developed for various data types, so DIF is empowered to handle diverse data types by plugging the data-specific network backbone (e.g., multi-perceptron networks, recurrent networks, or graph neural networks) to produce the representations (see Sec. 6.1.3 for different neural networks used in DIF and Sec. 6.2 for their performance in different data types).

5.5.2 Optimised vs. Casually Initialised Representations In general, we can use many representation learning networks specifically designed for anomaly detection, such as those in [10], [11], [13], [33], to obtain well-optimised feature representations. These representations are more expressive than randomly initialised representations if the optimisation objective well fits the input data. However, instead of using optimised representations, DIF uses the casually initialised representations due to the following two main reasons. (i) These loss functions are not versatile. It is difficult to devise one representation learning loss that can fit different data with diversified characteristics. (ii) The downstream data partition might be strongly controlled by the optimisation process. This way weakens the randomness and diversity of the feature representations, which are required in isolationbased anomaly scoring methods. These two intuitions are empirically investigated in our experiments by showing the performance of DIF on optimised representations produced by recent normality feature learning algorithms and the quality of these optimised representations (see Sec. 6.5.1)

5.5.3 Synergy between Random Representations and Random Partition-based Isolation DIF employs a novel representation scheme, i.e., the random representation ensemble produced via optimisationfree neural networks. The parameters of these networks can be initialised by randomly sampling from widely-used initialisation distributions (e.g., normal or uniform distribution), easily yielding a set of feature representations with excellent randomness and diversity. Given a sufficiently large set of such random representations, we can largely boost the isolation power in the random data partition, making it possible to effectively isolate some really hard anomalies on some subsets of these representations. For

Original Data Space
Projected Data Space
Projected Data Space
Projected Data Space

Fig. 3. Data space transformation produced by casually initialised neural networks (original data space vs. three transferred data spaces).

example, as shown in Fig. 1, among a large set of new representation spaces, there are some selective new spaces where hard anomalies become easy-to-isolate data objects. Recall that isolation methods, including DIF, are based on an average measure for anomaly scoring. Thus, the anomalies would stand out in the anomaly scores as long as they are effectively isolated in some of the isolation trees. DIF utilises this unique synergy between random representations and random partition-based isolation to largely improve the isolation process and subsequently the anomaly scoring function, resulting in significantly improved effectiveness of isolation-based anomaly detection. We empirically investigate the significance of this synergy by respectively replacing random representations and random partitionbased isolation with multiple alternatives (see Sec. 6.5).

6
EXPERIMENTS

We now introduce our experimental analysis. This section is organised as follows: In Sec. 6.1, we first start with the experimental setup including the used datasets, competing methods, parameter settings, and evaluation metrics; Sec. 6.2, 6.3, and 6.4 evaluate the performance of our method w.r.t. effectiveness, scalability, and robustness; and in Sec. 6.5 and 6.6, we empirically analyse our method by investigating the significance of the synergy between random representations and random partition-based isolation and the contribution of CERE and DEAS.

6.1
Experimental Setup

6.1.1
Datasets

We employ a large collection of publicly available and commonly-used real-world datasets, including ten tabular datasets, four graph datasets, and four time-series datasets. Their basic information is shown in Table 2.

Tabular Data. Analysis, Backdoor, DoS, and Exploits are taken from a popular intrusion detection benchmark UNSW NB 15. Following [15], [34], we select different attacks as anomalies in these datasets against normal network traffic. R8 is a highly-imbalanced text classification dataset, where the rare class are treated as anomalies by following [10], [35]. Cover is from the ecology domain. Fraud is for fraudulent credit card transaction detection. Pageblocks and Shuttle are provided by an anomaly benchmark study [36]. Thrombin is to detect unusual molecular bio-activity for drug design, which is an ultrahigh-dimensional anomaly detection dataset used in [10]

Graph Data. We employ datasets from the popular graph benchmark Tox21, which is a project on toxicity

<!-- page 8 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
8

TABLE 2
Dataset information of the used tabular, graph, and time series (TS)

datasets. N indicates the number of data objects, with parenthesis denoting the pre-defined training/testing size of graph and time series datasets. D denotes the number of dimensionality in tabular datasets,

the average number of nodes per graph in graph datasets, and the number of sequences in each time series dataset. #Anom (ratio) is the

number of anomalies and the corresponding ratios.

Data
N
D
#Anom (ratio)

Tabular

Analysis
95,677
197
2,677 (2.80%)
Backdoor
95,329
197
2,329 (2.44%)
DoS
96,000
197
3,000 (3.13%)
Exploits
96,000
197
3,000 (3.13%)
R8
3,974
9,468
51 (1.28%)
Cover
286,048
11
2,747 (0.96%)
Fraud
284,807
30
492 (0.17%)
Pageblocks
5,393
11
510 (9.46%)
Shuttle
1,013
10
13 (1.28%)
Thrombin
1,909
139,352
42 (2.20%)

Graph

HSE
8,417 (8,150/267)
17
10 (3.75%)
MMP
7,558 (7,320/238)
18
38 (0.50%)
p53
8,903 (8,634/269)
18
28 (10.41%)
PPAR
8,451 (8,184/267)
17
15 (5.62%)

TS

Mars
11,349 (3,500/7,849)
5
64 (0.82%)
Gait
65,000 (22,167/42,833)
3
401 (0.94%)
Heart
55,374 (20,185/35,189)
3
201 (0.57%)
Heart-w
80,000 (20,000/60,000)
1
251 (0.42%)

evaluation of newly synthesised or used chemical compounds. These datasets are chosen since they are inherently imbalanced and contain real anomalies. The task is to detect abnormal graphs.

Time-series Data. Time series data are taken from the UCR time series anomaly archive [37]. We employ data with natural anomalies. Mars is from NASA spacecraft. Gait is sensor data of a subject who has Huntington’s disease (highly asymmetric gait), and anomalies are data from the weak leg. ECG is heartbeat data, where anomalies are ventricular beats, and ECG-wandering (ECG-w for short) is a long stretch of ECG with a wandering baseline.

6.1.2
Competing Methods
To have a comprehensive comparison, DIF is compared with
the following two types of anomaly detection approaches.

iForest and Its Extensions (IF-based Methods). Apart from the popular iForest algorithm [3], its three advanced variants, i.e., EIF [4], PID [6], and LeSiNN [22], are employed. EIF slices data by using hyperplanes with random slopes and intercepts. In PID, the choice of splits is optimised based on the variance in the sparsity, and the anomaly scoring is redefined according to the sparsity. LeSiNN employs the nearest neighbour distance-based isolation ensemble, which is also known as aNNE in [38].

Ensemble of Deep Anomaly Detectors. 1. Different state-of-the-art (SOTA) deep anomaly detection methods are employed as base models, and we use the deep ensemble framework [26] to construct a suite of ensemble-based deep contenders. For tabular data, we utilise four stateof-the-art deep methods including RDP [13], REPEN [10], Deep SVDD [11], and a reconstruction-based Autoencoder baseline (RECON for short) [1]. We also employ the CERE ensemble method used in DIF for those methods to ensure their time efficiency and have a fair competition. As for

1. The ensemble performance of these deep models generally outperforms their solitary versions, and thus we focus on the comparison between ensemble-based results in the following experiments.

graph data, a deep graph-level anomaly detector GLocalKD [39] is used. TranAD [40] is employed for time-series data. Their ensemble versions are denoted as eRDP, eREPEN, eDSVDD, eRECON, eGLocalKD, and eTranAD. All these methods are specifically designed for anomaly detection on the corresponding data type.

6.1.3 Parameter Settings and Implementations DIF uses 50 representations (r=50) and 6 isolation trees per representation (t=6), with 256 as subsampling size (n=256) for each iTree. DIF processes tabular data by using fullyconnected multi-layer-perceptron networks. All of the IF- based competing methods use 300 trees (the ensemble size is the same as DIF). The subsampling size is set as 256. We use the maximum extension level of EIF, i.e., the extension level is adaptively set as the dimensionality minus 1. For LeSiNN, the subsampling size is 8 by following [22]. As IF- based competitors cannot directly process non-tabular data, we employ the latest powerful unsupervised representation learning models to yield high-quality vectorised representations, and specifically, InfoGraph [41] and TS2Vec [42] are respectively utilised for graph data and time series. Note that InfoGraph uses GIN as its graph encoder architecture, and TS2Vec applies a dilated CNN module with residual blocks. They propose specific learning objectives to optimise generated representations. For the sake of fairness, DIF also respectively utilises the same GIN and CNN network structure to handle graph data and time series but without any optimisation steps. The deep anomaly detectors are trained by 50 epochs and 30 steps per epoch. REPEN, DSVDD, and RECON take an Adam optimiser with a 1e-3 learning rate and use 64 objects per mini-batch, and we empirically found that RDP can work significantly better when using 1e-4. The default/recommended settings are used for the other parameters of these competing methods.

All the anomaly detection algorithms in our experiments are implemented using Python, with iForest from scikit-learn package, EIF from eif package, and other methods from their authors’ releases. The implementation of our method is publicly available2.

6.1.4 Evaluation Metrics and Computing Infrastructure Following the mainstream evaluation protocols of anomaly detection [3], [4], [15], the detection accuracy is evaluated by two complementary metrics including Area Under the ROC Curve (AUC-ROC) and Area Under the PR Curve (AUC-PR). ROC curve indicates the true positives against false positives, while PR curve summarises precision and recall of the anomaly class only. The paired Wilcoxon signed rank test is used to examine the statistical significance of the performance of DIF against each competing method.

We then introduce a new metric called Anomaly Isoability Index (AII) to measure the quality of representations. We borrow the concept of triplet loss [43] to count the percentage of effectively isolated anomalies among all true anomalies in each representation space, i.e.,

AII = P

a∼A



median

ni∈N

 1

|C|

X

nj∈C

 ω(a|ni, nj)

 > 0



,
(9)

2. Source code of DIF can be downloaded from https://github.com/ xuhongzuo/deep-iforest

<!-- page 9 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
9

TABLE 3 AUC-ROC and AUC-PR performance (mean ± standard deviation) of DIF and IF-based competing methods on ten real-world tabular datasets.

PID and EIF runs out of memory (OOM) on the ultrahigh-dimensional dataset Thrombin. The best performer is boldfaced.

Data
AUC-ROC
AUC-PR
DIF (ours)
EIF
PID
LeSiNN
IF
DIF (ours)
EIF
PID
LeSiNN
IF
Analysis
0.931±0.006
0.910±0.005
0.820±0.019
0.903±0.008
0.782±0.017
0.404±0.051
0.198±0.022
0.075±0.007
0.183±0.028
0.063±0.006
Backdoor
0.918±0.002
0.902±0.005
0.808±0.016
0.894±0.006
0.731±0.021
0.453±0.051
0.218±0.028
0.066±0.005
0.205±0.031
0.046±0.004
DoS
0.932±0.003
0.918±0.004
0.802±0.013
0.896±0.009
0.747±0.020
0.440±0.023
0.269±0.027
0.075±0.004
0.185±0.028
0.060±0.005
Exploits
0.858±0.010
0.840±0.008
0.797±0.011
0.816±0.005
0.745±0.010
0.273±0.020
0.167±0.011
0.077±0.003
0.120±0.013
0.062±0.003
R8
0.930±0.008
0.854±0.006
0.881±0.018
0.859±0.001
0.853±0.016
0.145±0.031
0.101±0.009
0.078±0.011
0.094±0.000
0.075±0.008
Cover
0.972±0.010
0.872±0.017
0.939±0.007
0.885±0.008
0.888±0.017
0.246±0.069
0.040±0.006
0.069±0.006
0.051±0.004
0.055±0.008
Fraud
0.953±0.002
0.950±0.001
0.950±0.002
0.952±0.000
0.950±0.001
0.387±0.031
0.378±0.027
0.186±0.033
0.401±0.001
0.155±0.015
Pageblocks
0.903±0.006
0.902±0.001
0.851±0.003
0.887±0.002
0.900±0.005
0.547±0.012
0.537±0.006
0.421±0.011
0.511±0.007
0.476±0.013
Shuttle
0.941±0.006
0.843±0.009
0.864±0.017
0.805±0.005
0.862±0.019
0.150±0.017
0.061±0.003
0.059±0.008
0.048±0.001
0.075±0.014
Thrombin
0.913±0.003
OOM
OOM
0.912±0.000
0.905±0.002
0.468±0.020
OOM
OOM
0.458±0.001
0.372±0.008
Average
0.925±0.006
0.888±0.006
0.857±0.011
0.881±0.004
0.836±0.013
0.351±0.033
0.219±0.015
0.123±0.010
0.226±0.011
0.144±0.008
p-value
-
0.004
0.004
0.002
0.002
-
0.004
0.004
0.006
0.002

TABLE 4 AUC-ROC and AUC-PR performance of DIF and its deep ensemble-based competing methods.

Data
AUC-ROC
AUC-PR
DIF (ours)
eRDP
eREPEN
eDSVDD
eRECON
DIF (ours)
eRDP
eREPEN
eDSVDD
eRECON
Analysis
0.931±0.006
0.899±0.005
0.883±0.026
0.844±0.006
0.862±0.002
0.404±0.051
0.400±0.019
0.168±0.051
0.374±0.028
0.093±0.002
Backdoor
0.918±0.002
0.900±0.006
0.863±0.022
0.916±0.004
0.840±0.003
0.453±0.051
0.422±0.020
0.163±0.055
0.448±0.030
0.082±0.004
DoS
0.932±0.003
0.905±0.004
0.861±0.025
0.900±0.004
0.856±0.004
0.440±0.023
0.420±0.013
0.154±0.047
0.409±0.028
0.105±0.004
Exploits
0.858±0.010
0.795±0.008
0.744±0.040
0.748±0.017
0.808±0.002
0.273±0.020
0.250±0.008
0.080±0.019
0.249±0.013
0.086±0.001
R8
0.930±0.008
0.864±0.020
0.904±0.004
0.883±0.014
0.779±0.000
0.145±0.031
0.123±0.017
0.086±0.005
0.074±0.006
0.082±0.000
Cover
0.972±0.010
0.966±0.008
0.896±0.013
0.981±0.005
0.922±0.004
0.246±0.069
0.192±0.049
0.044±0.006
0.353±0.055
0.067±0.003
Fraud
0.953±0.002
0.947±0.002
0.954±0.001
0.948±0.002
0.946±0.001
0.387±0.039
0.329±0.025
0.345±0.030
0.480±0.026
0.372±0.002
Pageblocks
0.903±0.010
0.850±0.005
0.896±0.005
0.882±0.005
0.829±0.012
0.547±0.020
0.466±0.011
0.511±0.019
0.397±0.009
0.542±0.032
Shuttle
0.941±0.006
0.900±0.009
0.780±0.025
0.837±0.037
0.779±0.014
0.150±0.017
0.132±0.009
0.031±0.003
0.144±0.022
0.044±0.002
Thrombin
0.913±0.003
0.869±0.019
0.914±0.004
0.366±0.019
0.911±0.000
0.468±0.020
0.368±0.053
0.399±0.056
0.016±0.001
0.457±0.000
Average
0.925±0.006
0.889±0.009
0.870±0.016
0.831±0.011
0.853±0.004
0.351±0.034
0.310±0.022
0.198±0.029
0.294±0.022
0.193±0.005
p-value
-
0.002
0.01
0.01
0.002
-
0.002
0.002
0.232
0.002

where ω(a|ni, nj) = d(a, ni)−d(nj, nj) denotes the difference between the Euclidean distances of the two pairs, a∼A represents any anomaly drawn from the true anomaly set in the dataset, C is a group of randomly sampled normal anchors, and N is another set of random normal samples to delegate the whole normal distribution. All the above data objects are from the target representation space. |C|=20 and |N|=1000 are used as we found empirically that these two settings are sufficiently large to compute the AII metric.

The computational time of all methods is based on a workstation with Intel Xeon Silver 4210R CPU, a single NVIDIA TITAN RTX GPU, and 64 GB RAM.

6.2
Effectiveness in Reducing False Negatives

6.2.1 Tabular Data Table 3 and 4 present the AUC-ROC and AUC-PR results of our method DIF and eight competing methods. Overall, DIF largely reduces the false negatives compared to iForest and its three extensions, resulting in substantial averaged performance improvement in detection precision and/or recall rates, and thus, DIF obtains superior AUC-PR and AUC-ROC performance. Particularly, in the average AUC- PR, DIF significantly outperforms EIF (61%), PID (186%), LeSiNN (56%), iForest (144%), eRDP (13%), eREPEN (77%), eDSVDD (19%), and eRECON (82%). DIF also obtains 4% - 11% AUC-ROC improvement across these methods. As shown in Table 3, DIF significantly outperforms isolation-based methods at the 99% confidence level. DIF is the best isolation-based detector across all the ten datasets except AUC-PR on Fraud, on which the nearest neighbourbased anomaly measure LeSiNN is more effective. The

features in Fraud are the results of PCA transformation due to the confidentiality issues, and thus the distance concept used in the nearest neighbour information can well reflect the proximity relationship. By contrast, DIF does not rely on such prior information that is not always reliable in all the datasets. These results demonstrate superior isolation power of DIF, which can effectively isolate anomalies that may not be possible in existing IF-based methods due to the challenges like data sparsity and non-linearity. This is particularly true on challenging high-dimensional datasets like R8, Analysis, Backdoor and DoS.

Compared to deep ensemble-based methods, DIF performs significantly better at the 99% confidence level according to the AUC-ROC performance in Table 4. DIF achieves the best performance on seven out of ten datasets in terms of both AUC-ROC and AUC-PR, and it obtains very competitive results on the rest three datasets with less than 0.01 difference in AUC-ROC. Nevertheless, the comparison results between DIF and its deep ensemble-based counterparts are very encouraging given the fact that DIF does not involve any optimisation while these deep methods need to be properly trained using pre-defined objective functions to better expose anomalies. The superiority of DIF in this comparison owes to the representation diversity of each ensemble member and the unique synergy between random representations and random partition-based isolation in the downstream anomaly scoring process. More importantly, DIF runs significantly faster than these deep ensemblebased contenders by around two orders of magnitude (see Sec. 6.3). Its superior computational efficiency endows DIF with stronger practicability in real-world applications.

<!-- page 10 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
10

TABLE 5
Results on detecting abnormal graphs and anomalies in time series.

Data
AUC-ROC
AUC-PR
DIF (ours)
EIF
LeSiNN
iForest
eGLocalKD
DIF (ours)
EIF
LeSiNN
iForest
eGLocalKD

Graph

HSE
0.737±0.013
0.715±0.014
0.702±0.001
0.697±0.014
0.593±0.002
0.094±0.005
0.088±0.004
0.084±0.000
0.082±0.004
0.054±0.000
MMP
0.715±0.006
0.663±0.012
0.666±0.000
0.667±0.018
0.675±0.001
0.260±0.006
0.216±0.006
0.217±0.000
0.219±0.011
0.233±0.001
p53
0.680±0.008
0.597±0.017
0.606±0.000
0.619±0.013
0.640±0.001
0.177±0.006
0.138±0.004
0.144±0.000
0.143±0.004
0.150±0.000
PPAR
0.701±0.013
0.716±0.005
0.711±0.000
0.733±0.009
0.643±0.001
0.127±0.008
0.173±0.006
0.165±0.001
0.208±0.012
0.086±0.000

DIF (ours)
EIF
LeSiNN
iForest
eTranAD
DIF (ours)
EIF
LeSiNN
iForest
eTranAD

TS

Mars
0.952±0.017
0.980±0.006
0.942±0.014
0.947±0.015
0.947±0.016
0.626±0.024
0.458±0.031
0.400±0.009
0.390±0.043
0.334±0.020
Gait
0.998±0.001
0.997±0.001
0.998±0.000
0.997±0.001
0.998±0.000
0.835±0.064
0.772±0.048
0.829±0.010
0.741±0.073
0.806±0.010
ECG
0.997±0.001
0.986±0.001
0.987±0.000
0.987±0.001
0.976±0.001
0.809±0.031
0.705±0.004
0.710±0.000
0.711±0.002
0.692±0.001
ECG-w
1.000±0.000
0.988±0.001
0.985±0.001
0.981±0.001
0.990±0.000
1.000±0.000
0.255±0.011
0.219±0.008
0.181±0.005
0.297±0.001

6.2.2
Graph Data and Time Series

As reported in Sec. 6.1.3, traditional isolation-based methods are performed upon vectorised representations learned by unsupervised representation learning method InfoGraph [41] and TS2Vec [42]. DIF respectively utilises the same network structure (GIN and dilated CNN) in InfoGraph and TS2Vec but without any optimisation steps. GLocalKD [39] is a graph-level anomaly detector, and TranAD [40] is a deep anomaly detection method specifically designed for time series. We also employ the ensemble version of these two SOTA methods as competitors.

Detection performance on graph data and time series are shown in Table 5, in which EIF and LeSiNN are selected as contenders due to their preferable performance in tabular datasets over PID, with iForest also included as a baseline. DIF is the best performer on three out of four datasets in both graph-level and time series anomaly detection tasks. This performance of DIF is remarkable in that it outperforms not only the isolation-based methods that are empowered by the latest powerful representation learning models but the recent SOTA methods that are specifically designed to extensively learn data-type-specific characteristics (e.g., holistic graph structure or temporal dependence) for effectively detecting these graph/sequential anomalies. By contrast, DIF performs well in a unified framework by only replacing the network structure with a different randomly initialised network backbone, offering a significantly simpler yet data-type-agnostic effective solution.

We further show how DIF works by visualising its results on a time series dataset. Fig. 4 visualises the ECG- w data (the id is sddb49 in the UCR benchmark) and the detection results of DIF and its competing methods. As illustrated in the introduction of the benchmark, this dataset is a challenging case that may confuse many algorithms due to its wandering shape. Albeit wandering baseline in testing data, we can also see this in the training set, and thus detection models are expected to tolerate these noisy regions. Our method DIF reduces the false negatives by successfully yielding significantly higher anomaly scores on anomalous heartbeats highlighted in golden yellow. In comparison, competing methods are misled by the fluctuating trends, showing much higher scores on noisy regions and omitting real anomalies.

6.3
Scalability to High-dimensional, Large-scale Data

This experiment examines the scalability of DIF and its contenders. These anomaly detectors are performed on a

Train

Data
Test

Data
Scores

TranAD

Scores

iForest

Scores

LeSiNN

Scores

EIF

Scores

DIF (ours)

Fig. 4. (Top to bottom) Snippets of training/testing data of ECG-w with wandering baseline and detection results, i.e., anomaly scores, of DIF and its competitors. Anomalous heartbeats are highlighted in golden yellow. Red dashed lines indicate the reported highest anomaly scores in this anomalous duration.

16
64
256
1,024
4,096
Dimensionality

100

102

104

106

Training Time (in seconds)

1,000
4,000 16,000 64,000256,000
Data Size

100

102

104

106
DIF (cpu)
eRDP (cpu)
eREPEN (cpu)
eDSVDD (cpu)
eRECON (cpu)
EIF
PID
LeSiNN
iForest

16
64
256
1,024
4,096
Dimensionality

100

102

104

106

Training Time (in seconds)

1,000
4,000 16,000 64,000256,000
Data Size

100

102

104

106

DIF (gpu)
eRDP (gpu)
eREPEN (gpu)
eDSVDD (gpu)
eRECON (gpu)

Fig. 5. Scalability test results. (Top) The training time of all the anomaly detectors on a CPU device; and (Bottom) The results of deep ensemble-based methods (including DIF) on a GPU device.

group of synthetic tabular datasets with different sizes and dimensionalities to record their training times. Nine datasets are with 5,000 data objects, and their dimensions range from 16 to 4,096. The other nine datasets are with 32 features and varied sizes from a minimum of 1,000 up to 256,000. IF-based anomaly detectors only need CPU devices, while deep ensemble-based methods can leverage GPU acceleration. Therefore, we report the training time of DIF and its deep competitors on both GPU and CPU devices.

<!-- page 11 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
11

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0

AUC-ROC

Analysis

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Backdoor

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Dos

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Exploits

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
R8

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0

AUC-ROC

Cover

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Fraud

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Pageblocks

DIF
EIF
PID
LeSiNN
iForest

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Shuttle

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Thrombin

(a) DIF vs. IF-based Competing Methods

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0

AUC-ROC

Analysis

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Backdoor

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Dos

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Exploits

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
R8

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0

AUC-ROC

Cover

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Fraud

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Pageblocks

DIF
eRDP
eREPEN
eDSVDD
eRECON

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Shuttle

0%
5%
10%
0.5
0.6
0.7
0.8
0.9
1.0
Thrombin

(b) DIF vs. Deep Ensemble-based Competing Methods

Fig. 6. AUC-ROC w.r.t. different contamination ratios ρ (the percentage of anomalies in the training set).

Fig. 5 (top) shows the scalability test results of DIF and its competing methods on a CPU device, and Fig. 5 (bottom) reports the comparison using GPU. DIF and all the other isolation-based methods present good scalability w.r.t. both dimensionality and data size compared to deep ensemblebased methods when using CPU computation. This is owed to the subsampling-based methods applied to both data samples and dimensions in isolation-based detectors. Besides, deep ensemble-based methods can greatly benefit from GPU acceleration when handling high-dimensional data. DIF shows outstanding time efficiency compared to its deep ensemble-based counterparts since it only requires one feed-forward step instead of a large number of training epochs. These results demonstrate that DIF inherits excellent computational efficiency from the iForest. Note that DIF can obtain almost the same scalability as iForest w.r.t. the data dimensionality, as DIF performs isolation on newly projected spaces of much smaller dimensionality while iForest works on the original data space.

6.4
Robustness w.r.t. Anomaly Contamination

This experiment examines the performance of DIF and its contenders when datasets containing different anomaly contamination ratios. Time-series datasets and graph datasets have pre-defined train-test split and their training sets do not contain anomalies, and thus we use tabular datasets as our bases here. Following [15], [34], we adjust the contamination ratios by injecting/removing anomalies, such that ratios range from 0% to 10%. These detection approaches are trained on adjusted datasets with controlled contamination ratios and tested on the original version.

The AUC-ROC performance is reported in Fig. 6. Generally, the performance of all the anomaly detectors downgrades with the increasing contamination ratio. Nevertheless, DIF has relatively clear superiority and stronger robustness in most of the datasets. The success of deep ensemble on out-of-distribution robustness has been proved in many recent studies [27], [29], [44], which partially explains why deep ensemble-based methods show better robustness w.r.t. anomaly contamination than IF-based methods. However, these competing methods still fail to provide consistently good robustness (e.g., eRDP, eREPEN, and eDSVDD on R8 and eRECON on Backdoor). It is mainly due to two reasons: (i) their ensemble processes may suffer from the diversity problem, and (ii) their scoring strategies and training objec-

tives rely on strong assumptions such as the distance concept that might not hold in some datasets. It is interesting to note that eREPEN shows an uptrend in Shuttle. REPEN uses LeSiNN to estimate initial anomaly scores when generating triplet mini-batches. As LeSiNN is with good robustness on Shuttle, this initial estimation can reliably obtain more anomaly examples as positive data when the contamination ratio increases, and thus the triplet learning process might benefit from the augmentation of the positive class.

6.5
Significance of the Synergy between Random Rep-
resentations and Random Partition-based Isolation

We respectively replace random representations and random isolation-based anomaly scoring in DIF to investigate their synergy effect.

6.5.1
Representation Scheme

This experiment evaluates the effectiveness of our novel representation scheme by comparing it with representations produced by optimised neural networks, including RDP [13], REPEN [10], DSVDD [11] and Reconstruction-based Autoencoder [1]. That is, we replace random representations with representations learned by one of these methods, with all the other components of DIF fixed. These four variants are denoted as RDP-DIF, REPEN-DIF, DSVDD-DIF and RECON-DIF. RDP, REPEN, and DSVDD are originally designed for learning a good representation for anomaly detection, and a dense representation can be implicitly derived from Autoencoders. The representations are evaluated by the AUC-ROC performance and the individual representation quality as follows.

AUC-ROC Results. The AUC-ROC results are shown in the upper half of Fig. 7. DIF outperforms four optimised representation ensemble-based methods on five datasets and shows very competitive performance to the best performer on the other five datasets. Averagely, the random representation ensemble contributes to 5%, 5%, 7%, and 15% AUC-ROC improvement than RDP-DIF, REPEN-DIF, DSVDD-DIF, and RECON-DIF, respectively. Although DIF may not be the best performer on all the datasets, it is very encouraging to see this performance achieved by the ensemble of random representations when compared to those optimised representations trained by state-of-the-art learning objectives.

<!-- page 12 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
12

Analysis

Backdoor

Dos

Exploits

R8

Cover

Fraud

Pageblocks

Shuttle

Thrombin

100%
75%
50%
25%
0%
0.3
0.5

0.7

0.9

AII AUC-ROC

DIF (ours)
RDP-DIF

REPEN-DIF
DSVDD-DIF

RECON-DIF

Fig. 7. (Top) AUC-ROC of DIF and its variants that use optimised representations, and (Bottom) the quality (measured by AII) distribution of randomised representations used in DIF and representations optimised by RDP, REPEN, DSVDD, and RECON.

Quality per Representation. To further analyse the mechanism behind the above results, we directly evaluate the quality of each representation produced by DIF and its variants. The representation quality is measured by Anomaly Isoability Index (AII), as introduced in Sec. 6.1.4. The AII results of each representation used in five anomaly detection methods are shown in Fig. 7 (bottom), in which we use box plots to present the quality distribution of representations produced by 50 representations in the ensemble framework.

Based on the above experiment results, the following
three remarks can be made.

• Our representation scheme achieves desired diversity and randomness, while simultaneously maintaining stable expressiveness in each representation, enabling excellent synergy with the downstream isolation-based anomaly scoring mechanism. This is the main driving force behind the superior performance of DIF.

• Optimised representations can be with consistently good quality on some datasets (e.g., near 80% true anomalies are well isolated by RECON-DIF on Analysis, Backdoor, and DoS), whereas the lack of diversity in representations downgrades the efficacy of this ensemble framework.

• Optimisation may even lead to worse representations on some datasets compared to random representations (e.g., R8, Cover, Pageblocks and Thrombin). This may be due to the fact that the underlying assumption (e.g., one-class assumption) in their learning objectives may not hold in those datasets.

6.5.2 Scoring Strategy As shown in the prior experiment, random representations are with good diversity and stable quality, fostering a unique excellent synergy effect in downstream ensemblebased anomaly scoring. This section further justifies this intuition by investigating the effectiveness of combining random representations with other anomaly scoring methods, including probability-based anomaly scoring method ECOD [45], distance-based method KNN [46], and densitybased method LOF [47]. Each of these methods is used to replace the isolation-based scoring process, with all the other modules fixed. These variants are denoted as DIF- ECOD, DIF-KNN, and DIF-LOF, respectively. Similarly, we

Analysis

Backdoor

Dos

Exploits

R8

Cover

Fraud

Pageblocks

Shuttle

Thrombin

1.0

0.8

0.6

0.4
0.4

0.6

0.8

1.0

AUC-ROC AUC-ROC

DIF (ours)
DIF-ECOD
DIF-KNN
DIF-LOF

Fig. 8. (Top) AUC-ROC of DIF and its variants of using other anomaly scoring strategies upon random representations, and (Bottom) their effectiveness (measured by AUC-ROC) on each ensemble member within the randomised representation groups.

evaluate their AUC-ROC performance and the individual
scoring quality as follows.

AUC-ROC Results. The AUC-ROC results are shown in the upper half of Fig. 8. DIF outperforms these competing variants on seven out of ten datasets. Averagely, DIF outperforms DIF-ECOD, DIF-KNN, and DIF-LOF by 3%, 13%, and 25%, respectively. The superiority further justifies the synergy effect in DIF. Note that KNN and LOF are with very heavy computational overhead. They take around 60 hours to handle large-scale datasets Cover and Fraud.

Quality per Anomaly Scoring Result. We also study the quality of individual anomaly scoring results produced by DIF and its variants on each random representation, as shown in Fig. 8 (bottom). The quality is also estimated by AUC-ROC here.

We make the following three remarks.

• These competing scoring methods can produce markedly better individual scoring results than our isolation-based scoring mechanism on Exploits, R8, and Thrombin. However, they only yield less effective or slightly better integrated results compared to DIF. These scoring methods are unable to leverage the diversity embedded in our representation scheme.

• By contrast, DIF combines data representation and anomaly scoring in a successful unified ensemble learning framework. DIF achieves better integrated performance by fully leveraging the diversity and randomness of representations, e.g., above 0.9 AUC-ROC on R8 with the maximum individual value only achieving around 0.7.

• DIF is inferior to its variants on Shuttle. It might be because anomalies in this dataset can be more easily identified by using the prior concepts used in these competing scoring methods (i.e., probability, distance, or density). However, these prior concepts may fail to work properly across all the datasets.

6.6
Ablation Study on CERE and DEAS

Considering detection effectiveness and computational efficiency, we propose the Deviation-Enhanced Anomaly Scoring function (DEAS) and the Computation-Efficient deep Representation Ensemble method (CERE) in the specific implementation of DIF. This experiment is conducted to examine whether DIF can have better detection performance

<!-- page 13 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
13

TABLE 6
AUC-ROC and AUC-PR results of DIF and w/o DEAS, and training time

(in seconds) of DIF and w/o CERE. w/o DEAS is an ablated version by

replacing DEAS with the standard scoring function. w/o CERE only

uses the typical method to produce the representation ensemble.

Data
AUC-ROC
AUC-PR
Time (in seconds)
DIF
w/o DEAS
DIF
w/o DEAS
DIF
w/o CERE
Analysis
0.931
0.922±0.007
0.404
0.315±0.058
10.5
69.2
Backdoor
0.918
0.914±0.006
0.453
0.381±0.067
10.5
66.2
DoS
0.932
0.926±0.007
0.440
0.388±0.074
10.4
65.6
Exploits
0.858
0.854±0.008
0.273
0.248±0.041
10.4
65.8
R8
0.930
0.915±0.012
0.145
0.123±0.034
5.2
19.2
Cover
0.972
0.964±0.014
0.246
0.210±0.084
33.0
154.3
Fraud
0.953
0.952±0.002
0.387
0.387±0.042
33.6
169.1
Pageblocks
0.903
0.912±0.007
0.547
0.576±0.024
0.8
3.5
Shuttle
0.941
0.923±0.012
0.150
0.103±0.019
0.5
1.0
Thrombin
0.913
0.916±0.002
0.468
0.448±0.020
13.8
336.7
Avg./Total
0.925
0.920±0.008
0.351
0.318±0.046
128.7
950.6
p-value
-
0.07
-
0.03
-
-

and use less training time with the help of DEAS and CERE. Two ablated variants are employed, i.e., w/o CERE replaces GCERE with the conventional sequential ensemble process, and w/o DEAS uses the standard scoring function used in iForest to replace our scoring function FDEAS.

The AUC-ROC and AUC-PR results of DIF and w/o DEAS are shown in Table 6. DIF significantly outperforms w/o DEAS at the 90% confidence interval and achieves approximate 11% AUC-PR improvement. Besides, DIF costs considerably less training time with the help of CERE. The total training time across all the ten datasets is only approximate one-tenth of the variant w/o CERE. The AUC- ROC/AUC-PR results of w/o CERE are on par with DIF, which are omitted due to the space limitation. Based on the above comparison results, the contribution of the proposed DEAS and CERE is validated and quantitatively measured.

7
CONCLUSIONS

This paper introduces DIF, a novel extension of iForest. DIF takes the deep neural network-based random representation ensemble as a new representation scheme, enabling significantly versatile data partition in diverse random directions on subspaces of different sizes. The anomaly scoring can be facilitated by the synergy between random representations and random partition-based isolation. This enables DIF to fulfil (i) more effective isolation of anomalies, especially hard anomalies in data with intractable sparsity and nonlinearity; (ii) liberation of isolation process from existing constraints to tackle the artefact problem; and (iii) versatile ability to handle different data types. Extensive experiments show that DIF significantly outperforms iForest and its existing extensions not only on tabular data but also on graph and time-series data. DIF also shows promising improvement compared to the ensemble of state-of-the-art deep anomaly detectors.

In our future work, we plan to devise new scoring strategies, together with relevant neural network backbones, to address other challenging yet important anomaly detection tasks, e.g., identify possible abnormal subsets [48] and multi-view data [49].

ACKNOWLEDGMENTS

Hongzuo Xu, Yijie Wang, and Yongjun Wang are supported by the National Key R&D Program of China (No.

2022ZD0115302), the National Natural Science Foundation of China (No.61379052), the Science Foundation of Ministry of Education of China (No.2018A02002), the Postgraduate Scientific Research Innovation Project of Hunan Province (CX20210049), the Natural Science Foundation for Distinguished Young Scholars of Hunan Province (No.14JJ1026). Guansong Pang is supported in part by the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant (21SISSMU031).

We thank Eamonn Keogh and his students for creating the UCR time series anomaly archive that has been used in our experiment. We also thank the referees for their comments, which helped improve this paper considerably.”

REFERENCES

[1]
C. C. Aggarwal, Outlier analysis.
Springer, 2017.
[2]
G. Pang, C. Shen, L. Cao, and A. V. D. Hengel, “Deep learning for
anomaly detection: A review,” ACM Comput. Surv., vol. 54, no. 2,
pp. 1–38, 2021.
[3]
F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in Proc.
8th IEEE Int. Conf. Data Mining.
IEEE, 2008, pp. 413–422.
[4]
S. Hariri, M. C. Kind, and R. J. Brunner, “Extended isolation
forest,” IEEE Trans. Knowl. Data Eng., vol. 33, no. 4, pp. 1479–1489,
2021.
[5]
F. T. Liu, K. M. Ting, and Z.-H. Zhou, “On detecting clustered
anomalies using sciforest,” in European Conf. Mach. Learn. Principles
Practice Knowl. Discovery Databases.
Springer, 2010, pp. 274–290.
[6]
P. Gopalan, V. Sharan, and U. Wieder, “Pidforest: anomaly de-
tection via partial identification,” in Proc. Adv. Neural Inf. Process.
Syst., 2019, pp. 15 809–15 819.
[7]
J. Lesouple, C. Baudoin, M. Spigai, and J.-Y. Tourneret, “General-
ized isolation forest for anomaly detection,” Pattern Recognit. Lett.,
2021.
[8]
D. Cortes, “Revisiting randomized choices in isolation forests,”
arXiv preprint arXiv:2110.13402, 2021.
[9]
M. Tokovarov and P. Karczmarek, “A probabilistic generalization
of isolation forest,” Inf. Sci., vol. 584, pp. 433–449, 2022.
[10] G. Pang, L. Cao, L. Chen, and H. Liu, “Learning Representations of

Ultrahigh-dimensional Data for Random Distance-based Outlier Detection,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2018, pp. 2041–2050. [11] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui,

A. Binder, E. M¨uller, and M. Kloft, “Deep one-class classification,” in Proc. Int. Conf. Mach. Learn. PMLR, 2018, pp. 4393–4402. [12] L. Bergman and Y. Hoshen, “Classification-based anomaly detec-

tion for general data,” in Proc. Int. Conf. Learn. Represent., 2020. [13] H. Wang, G. Pang, C. Shen, and C. Ma, “Unsupervised represen-

tation learning by predicting random distances,” in Proc. Int. Joint Conf. Artif. Intell., 2021, pp. 2950–2956. [14] B. Liu, D. Wang, K. Lin, P.-N. Tan, and J. Zhou, “Rca: A deep

collaborative autoencoder approach for anomaly detection,” in Proc. Int. Joint Conf. Artif. Intell., 2021. [15] G. Pang, C. Shen, and A. van den Hengel, “Deep anomaly detec-

tion with deviation networks,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2019, pp. 353–362. [16] L. Ruff, J. R. Kauffmann, R. A. Vandermeulen, G. Montavon,

W. Samek, M. Kloft, T. G. Dietterich, and K.-R. M¨uller, “A unifying review of deep and shallow anomaly detection,” Proc. IEEE, 2021. [17] A. F. Emmott, S. Das, T. Dietterich, A. Fern, and W.-K. Wong,

“Systematic construction of anomaly detection benchmarks from real data,” in Proc. ACM SIGKDD Workshop Outlier Detection Description, 2013, pp. 16–21. [18] K.-H. Lai, D. Zha, G. Wang, J. Xu, Y. Zhao, D. Kumar, Y. Chen,

P. Zumkhawaka, M. Wan, D. Martinez et al., “Tods: An automated time series outlier detection system,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 18, 2021, pp. 16 060–16 062. [19] S. Alnegheimish, D. Liu, C. Sala, L. Berti-Equille, and K. Veera-

machaneni, “Sintel: A machine learning framework to extract insights from signals,” in Proc. Int. Conf. Manag. Data, 2022, pp. 1855–1865. [20] Y. Zhao, Z. Nasrullah, and Z. Li, “Pyod: A python toolbox for

scalable outlier detection,” Journal of Machine Learning Research, vol. 20, pp. 1–7, 2019.

<!-- page 14 -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
14

[21] “Anomaly detection by ibm,” https://developer.ibm.com/apis/

catalog/ai4industry--anomaly-detection-product/Introduction/, (accessed Feb. 27, 2023). [22] G. Pang, K. M. Ting, and D. Albrecht, “Lesinn: Detecting anoma-

lies by identifying least similar nearest neighbours,” in Proc. 15th IEEE Int. Conf. Data Mining Workshops. IEEE, 2015, pp. 623–630. [23] T. R. Bandaragoda, K. M. Ting, D. Albrecht, F. T. Liu, Y. Zhu,

and J. R. Wells, “Isolation-based anomaly detection using nearestneighbor ensembles,” Comput. Intell., vol. 34, no. 4, pp. 968–998, 2018. [24] X. Zhang, W. Dou, Q. He, R. Zhou, C. Leckie, R. Kotagiri, and

Z. Salcic, “Lshiforest: A generic framework for fast tree isolation based ensemble anomaly analysis,” in Proc. Int. Conf. Data Eng. IEEE, 2017, pp. 983–994. [25] A. Mensi and M. Bicego, “Enhanced anomaly scores for isolation

forests,” Pattern Recognit., p. 108115, 2021. [26] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and

scalable predictive uncertainty estimation using deep ensembles,” Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017. [27] F. D’Angelo and V. Fortuin, “Repulsive deep ensembles are

bayesian,” Proc. Adv. Neural Inf. Process. Syst., vol. 34, pp. 3451– 3465, 2021. [28] A. Ram´e and M. Cord, “Dice: Diversity in deep ensembles via con-

ditional redundancy adversarial estimation,” in Int. Conf. Learn. Represent., 2021. [29] G. Nam, J. Yoon, Y. Lee, and J. Lee, “Diversity matters when

learning from ensembles,” Proc. Adv. Neural Inf. Process. Syst., vol. 34, pp. 8367–8377, 2021. [30] J. Chen, S. Sathe, C. Aggarwal, and D. Turaga, “Outlier detection

with autoencoder ensembles,” in Proc. SIAM Int. Conf. Data Min-
ing.
SIAM, 2017, pp. 90–98.
[31] T. Kieu, B. Yang, C. Guo, and C. S. Jensen, “Outlier detection for

time series with recurrent autoencoder ensembles.” in Proc. Int. Joint Conf. Artif. Intell., 2019, pp. 2725–2732. [32] Y. Wen, D. Tran, and J. Ba, “Batchensemble: an alternative ap-

proach to efficient ensemble and lifelong learning,” in Int. Conf. Learn. Represent., 2019. [33] H. Xu, Y. Wang, S. Jian, Z. Huang, Y. Wang, N. Liu, and F. Li,

“Beyond outlier detection: Outlier interpretation by attentionguided triplet deviation network,” in Proc. Web Conf., 2021, pp. 1328–1339. [34] G. Pang, A. van den Hengel, C. Shen, and L. Cao, “Toward

deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2021, pp. 1298–1308. [35] S. Rayana and L. Akoglu, “Less is more: Building selective

anomaly ensembles,” ACM Trans. Knowl. Discovery Data, vol. 10, no. 4, pp. 1–33, 2016. [36] G. O. Campos, A. Zimek, J. Sander, R. J. Campello, B. Micenkov´a,

E. Schubert, I. Assent, and M. E. Houle, “On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study,” Data Mining Knowl. Discovery, vol. 30, no. 4, pp. 891–927, 2016. [37] R. Wu and E. Keogh, “Current time series anomaly detection

benchmarks are flawed and are creating the illusion of progress,” IEEE Trans. Knowl. Data Eng., 2021. [38] K. M. Ting, T. Washio, J. R. Wells, and S. Aryal, “Defying the

gravity of learning curve: a characteristic of nearest neighbour anomaly detectors,” Mach. Learn., vol. 106, no. 1, pp. 55–91, 2017. [39] R. Ma, G. Pang, L. Chen, and A. v. d. Hengel, “Deep graph-level

anomaly detection by glocal knowledge distillation,” in Proc. Int. Conf. Web Search Data Mining, 2022. [40] S. Tuli, G. Casale, and N. R. Jennings, “TranAD: Deep Transformer

Networks for Anomaly Detection in Multivariate Time Series Data,” Proc. VLDB Endowment, vol. 15, no. 6, pp. 1201–1214, 2022. [41] F.-Y. Sun, J. Hoffman, V. Verma, and J. Tang, “Infograph: Unsuper-

vised and semi-supervised graph-level representation learning via mutual information maximization,” in Int. Conf. Learn. Represent, 2020. [42] Z. Yue, Y. Wang, J. Duan, T. Yang, C. Huang, Y. Tong, and B. Xu,

“Ts2vec: Towards universal representation of time series,” in Proc. AAAI Conf. Artif. Intell., vol. 36, no. 8, 2022, pp. 8980–8987. [43] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified

embedding for face recognition and clustering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 815–823. [44] S. Liu, T. Chen, Z. Atashgahi, X. Chen, G. Sokar, E. Mocanu,

M. Pechenizkiy, Z. Wang, and D. C. Mocanu, “Deep ensembling

with no overhead for either training or testing: The all-round blessings of dynamic sparsity,” in Int. Conf. Learn. Represent., 2022. [45] Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, and G. Chen, “Ecod:

Unsupervised outlier detection using empirical cumulative distribution functions,” IEEE Trans. Knowl. Data Eng., 2022. [46] S. Ramaswamy, R. Rastogi, and K. Shim, “Efficient algorithms for

mining outliers from large data sets,” in Proc. Int. Conf. Manag. Data, 2000, pp. 427–438. [47] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “Lof:

identifying density-based local outliers,” in Proc. Int. Conf. Manag. Data, 2000, pp. 93–104. [48] L. Huang, Y. Zhu, Y. Gao, T. Liu, C. Chang, C. Liu, Y. Tang,

and C.-D. Wang, “Hybrid-order anomaly detection on attributed networks,” IEEE Trans. Knowl. Data Eng., 2021. [49] Y.-X. Ji, L. Huang, H.-P. He, C.-D. Wang, G. Xie, W. Shi, and K.-Y.

Lin, “Multi-view outlier detection in deep intact space,” in Proc. 19th IEEE Int. Conf. Data Mining. IEEE, 2019, pp. 1132–1137.

Hongzuo Xu received the bachelor’s and master’s degree from the National University of Defense Technology, China, in 2017 and 2019, where he is currently pursuing the Ph.D. degree in computer science. His research interests include anomaly detection, outlier interpretation, weakly-supervised learning and data mining.

Guansong Pang is a tenure-track Assistant Professor of Computer Science in the School of Computing and Information Systems at Singapore Management University (SMU), Singapore. Before joining SMU, he was a Research Fellow with the Australian Institute for Machine Learning (AIML). He received a PhD degree from University of Technology Sydney in 2019. His research explores novel data mining and machine learning techniques and their applications, with a focus on detecting abnormal and unknown data.

Yijie Wang received the PhD degree in computer science and technology from the National University of Defense Technology in 1998. She was awarded the prize of National Excellent Doctoral Dissertation by Ministry of Education of PR China. She is currently a Full Professor with the Science and Technology on Parallel and Distributed Processing Laboratory (PDL), National University of Defense Technology. Her research interests include big data analysis, artificial intelligence and parallel and distributed processing.

Yongjun Wang received the Ph.D. degree in computer architecture from the National University of Defense Technology, China, in 1998. He is currently a Full Professor with the College of Computer, National University of Defense Technology, Changsha, China. His research interests include network security and system security.
