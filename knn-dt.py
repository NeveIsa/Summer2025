import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set_style("whitegrid")
    return mo, np, plt, sns


@app.cell(hide_code=True)
def _(np):
    def knn(point, params, k=5, weighted=False):
        # point is a numpy vector
        point = np.array(point)
        # params is a list of numpy matrices/2d-arrays and those matrices have rows containing samples for that class
        # the classes are 0,1,2,... based on the position of those matrices within params list
        nclasses = len(params)
        classes = []
        norms = []

        for i in range(nclasses):
            samples  = params[i]
            diff = samples - point
            cnorms = np.linalg.norm(diff, axis=1)
            classes += [i]*samples.shape[0]
            norms += cnorms.tolist()

        topk = np.argsort(norms)[:k]

        votes = dict([(i,0) for i in range(nclasses)])

        for i in topk:
            c = classes[i]
            d = norms[i]
            votes[c] += 1/d if weighted else 1

        sortedvotes = sorted(votes.items(), key=lambda x: x[1])
        # print(len(sortedvotes))
        return sortedvotes[-1][0]      
    return (knn,)


@app.cell(hide_code=True)
def _(np, nsamples, std):
    red = np.random.randn(nsamples.value,2)*std.value + [3,7]
    green = np.random.randn(nsamples.value,2)*std.value + [5,3]
    blue = np.random.randn(nsamples.value,2)*std.value + [7,7]
    return blue, green, red


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # K Nearest Neighbors Classification
    - There is a feature space $F$ and each sample from the dataset is associated with an element in $F$.
    - Let the set of classes be denoted by $C$.
    - Pick any distance metric $d$ in the feature space $F$.
    - Metric is a function $d:F\times F \rightarrow [0,\infty)$ such that it obeys the three following laws
    - 0. $d(x,y) \geq 0$
      1. $d(x,y) = 0 \iff x = y$
      2. $d(x,y) = d(y,x)$
      3. $d(x,z) \leq d(x,y) + d(y,z)$
    - There is a ground truth function $g: F \rightarrow C$
    - We want to learn this function $g$ by approximating it with a hypothesis $h : F \rightarrow C$
    - Since we are using KNN, the hypothesis $h$ is a function learned via KNN.
    - Let $n_i := n(x,i)$ be the $i^{th}$ neighbor of $x$ in the dataset $D = \{ x_1, x_2, ... x_m\}$.
    - Let $w_i = {d(x,n_i)}^{-1}$ if weighted kNN else $w_i = 1$
    - $h_k(x) := \arg\max_{c \in C} A(c)$ 
    - $A(c) := \sum_{i\; : \; g(n_i) = c} w_i$
    - For KNN Regression, redefine $A(c) := \sum_{i} \frac{w_i}{W} \cdot g(n_i)$ where $W = \sum_i w_i$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    nsamples = mo.ui.slider(10,30,step=1,show_value=True,label="nsamples")
    std = mo.ui.slider(1,2,step=0.25, show_value=True, label="std")
    k = mo.ui.slider(1,15,step=2,debounce=True, label="k", show_value=True)
    weighted = mo.ui.checkbox(label="weighted")
    nsamples,std,k,weighted
    return k, nsamples, std, weighted


@app.cell(hide_code=True)
def _(blue, green, k, knn, np, plt, red, sns, weighted):

    all = [ (x,y) for x in np.arange(0,10,0.1) for y in np.arange(0,10,0.1) ]
    allclass = [knn(a,[red,green,blue],k=k.value, weighted=weighted.value) for a in all]
    allcolors = np.array(["red","green","blue"])[allclass]

    plt.xlim(0,10)
    plt.ylim(0,10)
    sns.scatterplot(x=red[:,0],y=red[:,1],color="red")
    sns.scatterplot(x=green[:,0],y=green[:,1],color="green")
    sns.scatterplot(x=blue[:,0],y=blue[:,1],color="blue")

    sns.scatterplot(x=[a[0] for a in all], y=[a[1] for a in all], color=allcolors, marker='1', alpha=0.25)
    plt.title(f"k = {k.value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Observations using k-nearest-neighbors

    - Partitioning of feature space based on voting from nearest **k** neighbors.
    - The partition of the feature space is based on the position (feature space) of the training samples and their classes
    - Once the partition of feature space in accomplished, any new sample is classified based on the class of that region.
    - The partition in kNN is non-linear (but piecewise linear; for k=1, it is the Voronoi tesselation)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Motivation for Decision Trees

    ## Objective : Binary Classification

    - We have a bag of samples and they either belong to the +ve or the -ve class.
    - A bag containing samples from only one class is considered pure.
    - We have a measure of impurity/purity of the bag which captures the above notion of purity.
    - For a collection of bags, the impurity is the sum of impurities/purities of the individual bags.
    - We want to improve the purity of the bags by subdiving them into smaller bags. Can this be done?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Measures of Purity/Impurity

    - A bag containing samples from only one class must have the highest/lowest purity in any measure of purity/impurity
    - Let $p$ be the fraction of the positive class in the bag, i.e the random chance of any picked sample is positive.
    - If $p=1$, we want the purity to be highest. The same goes if $p=0$ (why?). At both these locations, the highest purity value should be the same because there is no reason why we should have completely positive samples more/less pure than a completely negative samples.
    - Hence, as a function of p, the purity should be maximum at p=0 and p=1.
    - Purity should be minimum at p=0.5 (why?)

    |Name| Genreal Function|Function for Binary Classification|Remark|
    |:---|:---------------|:--------------------------------|:-------|
    |Classification Purity| $\max_i p_i$ | $\max\{p, 1-p\}$ |Convex, maxval = 1, range(0,1)|
    |Gini Purity| $\sum_i p_i^2$| $p^2 + (1-p)^2$| Convex, maxval=1, range(0,1)|
    |Shannon Purity/neg_entropy | $\sum_i p_i \log(p_i)$| $p \log(p) + (1-p)\log(1-p)$ |Convex, maxval=0, range($-\infty$, 0)|
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def classification_purity(p):
        p=np.array(p)
        return np.max(p)
    def gini_purity(p):
        p = np.array(p)
        return np.sum(p**2)
    def negentropy_purity(p):
        p = np.array(p)
        with np.errstate(divide='ignore', invalid='ignore'):
            return p @ np.log2(p)
    return classification_purity, gini_purity, negentropy_purity


@app.cell(hide_code=True)
def _(classification_purity, gini_purity, negentropy_purity, np, plt, sns):
    _p = np.arange(0,1,0.01)
    _args = list(zip(_p,1-_p))


    plt.figure(figsize=(16, 4)) 
    # plt.ylabel("purity")
    # plt.xlabel("p")


    plt.subplot(1,3,1)
    plt.grid()
    sns.lineplot(x=_p, y = [classification_purity(a) for a in _args], label="classification_purity", color="red")

    plt.subplot(1,3,2)
    plt.grid()
    sns.lineplot(x=_p, y = [gini_purity(a) for a in _args], label="gini_purity", color="green")

    plt.subplot(1,3,3)
    plt.grid()
    sns.lineplot(x=_p, y = [negentropy_purity(a) for a in _args], label="negentropy_purity", color="blue")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #Splitting always improves purity

    ### Jensen's Inequality

    For any convex function $f$ and fractions $w_1,w_2 \geq 0$ with $w_1 + w_2 = 1$, we have
    $$f(w_1x_1 + w_2x_2) \leq w_1f(x_1) + w_2f(x_2)$$

    - Let there be a collection with total $T$ samples with $p$ fraction of them being from positive class.

    - Let us partition this collection into two collections where the first collection gets $w_1$ fraction of the samples and similarly $w_2$ for the other.

    - Let $p_1$ and $p_2$ be the fraction of positives and negatives in the first collection. Similarly $q_1,q_2$ for the second collection.

    - Constraints
    $$pT = w_1 p_1T + w_2 q_1 T \implies p = w_1p_1 + w_2 q_1$$

    $$(1-p)T = w_1 p_2 T + w_2 q_2  T \implies 1-p = w_1p_2 + w_2q_2$$ 

    - Hence  $$\color{red}f(p) \leq w_1f(p_1) + w_2f(q_1)$$ and $$\color{green}f(1-p) \leq w_1f(p_2) + w_2f(q_2)$$

    - Before splitting
    $$Purity\;before\;split= f(p) + f(1-p)$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - After splitting
    $$Purity\;after\;split = w_1 \bigg[f(p_1) + f(p_2)\bigg] + w_2 \bigg[f(q_1) + f(q_2)\bigg] 
    \\\;\\= \bigg[\underbrace{w_1f(p_1) + w_2f(q_1)}_{red\;ineq.}\bigg] + \bigg[\underbrace{w_1f(p_2)+w_2f(q_2)}_{green\;ineq.}\bigg]
    \\\;\\\leq f(p) + f(1-p) \\ || \\ Purity\;before\;split$$
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
